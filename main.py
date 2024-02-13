import argparse
import datetime
import math
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.parallel
import vgg
import sys
import numpy as np
from scipy.stats import levy_stable

from models import MultiLayerNN, Net
from exp_utils import get_data, accuracy, eval, get_convergence_criteria, get_accuracy_lower_bound

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=100000, type=int)
    parser.add_argument('--batch_size_train', '-b', default=100, type=int)
    parser.add_argument('--batch_size_eval', default=0, type=int,
                        help='must be equal to training batch size')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--iterate_noise', default=0, type=int, choices=[0,1,2,3], help="0 means no noise added. 1, 2, 3 corresponds to types 1, 2, 3.")
    parser.add_argument('--iterate_noise_alpha', default=1.9, type=float)
    parser.add_argument('--iterate_noise_scale', default=1.0, type=float)
    parser.add_argument('--iterate_noise_filter', '--inf', action='store_true', default=False)
    parser.add_argument('--iterate_noise_bias', action='store_true', default=False)
    parser.add_argument('--iterate_noise_lr_rescaling', action='store_true', default=True)
    parser.add_argument('--mom', default=0, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--eval_freq', default=200, type=int)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--dataset', default='mnist', type=str, help='ecg5000 | mnist | cifar10 | cifar100')
    parser.add_argument('--convergence_method', type=str, choices=['custom', 'dataset', 'none'], default='custom')
    parser.add_argument('--convergence_accuracy', type=float, default=95.)
    parser.add_argument('--convergence_loss', type=float, default=np.inf)
    parser.add_argument('--convergence_extra_iters', type=int, default=0)
    parser.add_argument('--path', default='./data', type=str)
    parser.add_argument('--seed', default=0, type=int)  
    parser.add_argument('--model', default='fcn', type=str)
    parser.add_argument('--criterion', default='NLL', type=str,                         help='NLL | linear_hinge')
    parser.add_argument('--width', default=100, type=int, help='width of fully connected layers')
    parser.add_argument('--depth', default=2, type=int, help='total number of hidden layers + input layer')
    parser.add_argument('--save_dir', default='results/', type=str)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--script', '--experiment_script', '-s', nargs="+", type=str, default="")

    args = parser.parse_args() #parsing arguments

    torch.manual_seed(args.seed) # setting seed

    # Saving script, if exists
    script = "".join(args.script)
    args.script = ""
    print(args)

    if not args.batch_size_eval:
        args.batch_size_eval = args.batch_size_train

    iterate_noise_scale = args.iterate_noise_scale
    if args.iterate_noise_lr_rescaling:
        iterate_noise_scale = iterate_noise_scale * args.lr**(1/args.iterate_noise_alpha)
    
    loss_crit, accuracy_crit = get_convergence_criteria(args.dataset, args.convergence_method, args.convergence_loss, args.convergence_accuracy)

    begin_time = time.time() #timer begun
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Not using CUDA.") if not args.use_cuda else None
    args.device = torch.device('cuda' if args.use_cuda else 'cpu') 

    train_loader, test_loader_eval, train_loader_eval, num_classes, input_dim = get_data(args) 

    if "vgg" in args.model: # TODO
        net = vgg.__dict__[args.model]().to(args.device)
        net.features = torch.nn.DataParallel(net.features)
    else:
        net = MultiLayerNN(input_dim=input_dim, width=args.width, depth=args.depth, num_classes=num_classes, activation=args.activation).to(args.device)

    print(net)

    opt = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.mom,
        weight_decay=args.wd)

    if args.criterion == 'NLL':
        crit = nn.CrossEntropyLoss(reduction='mean').to(args.device)

    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    circ_train_loader = cycle_loader(train_loader)

    training_history = []
    weight_grad_history = []

    evaluation_history_TEST = []
    evaluation_history_TRAIN = []
    evaluation_history_AVG = []
    evaluation_history_AVGTRAIN = []

    STOP = False
            
    convergence_first_observed_at = 0
    
    for i, (x, y) in enumerate(circ_train_loader):
        net.train()

        x, y = x.to(args.device), y.to(args.device)

        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)
        loss.backward()

        training_history.append([i, loss.item(), accuracy(out, y).item()])

        if args.alpha > 0:
            for group in opt.param_groups:
                gan = (args.lr / args.width) ** (1 / (1 - args.alpha))
                group['lr'] = args.lr * (i + (1 / gan)) ** (- args.alpha)

        # take the step
        opt.step()
  
        if (i % args.eval_freq == 0): #or (j == 0):
            print("## Iteration", i)
            # first record is at the initial point
            print('train eval')
            tr_hist, tr_outputs = eval(train_loader_eval, net, crit, args)
            print('test eval')
            te_hist, te_outputs = eval(test_loader_eval, net, crit, args)
            evaluation_history_TRAIN.append([i, *tr_hist])
            evaluation_history_TEST.append([i, *te_hist])     

        if i == args.iterations: # changed this to equality
            STOP = True

        if np.isnan(tr_hist[0]) or np.isnan(te_hist[0]):
            print('Training terminated due to divergence.\n')
            sys.exit()

        if (i > 100000) and (tr_hist[1] <= get_accuracy_lower_bound(num_classes)):
            print('Training terminated due to no increase in accuracy.\n')
            sys.exit()

        if ((tr_hist[0] < loss_crit) and (tr_hist[1] >= accuracy_crit)):
            print(f'Finishing training due to convergence, with loss < {loss_crit} and accuracy > {accuracy_crit}.\n')
            STOP = True
        
        if (not STOP) and args.iterate_noise:
            first_conv = True
            for param in net.parameters():
                if len(param.data.size()) == 4:
                    if first_conv:
                        first_conv = False
                        continue
                    if (args.iterate_noise == 1) and args.iterate_noise_filter:
                        #raise NotImplementedError
                        noise = levy_stable.rvs(alpha=args.iterate_noise_alpha, beta=0, scale=1, size=(param.data.size()[:1]))[:, np.newaxis, np.newaxis, np.newaxis]
                    elif (args.iterate_noise == 1) and (args.iterate_noise_alpha==2.0):
                        #raise NotImplementedError
                        noise = np.random.randn(param.data.size()[:1])[:, np.newaxis, np.newaxis, np.newaxis]
                    elif args.iterate_noise == 2:
                        noise = levy_stable.rvs(alpha=args.iterate_noise_alpha, beta=0, scale=1, size=param.data.size())
                    elif args.iterate_noise == 3:
                        noise = np.random.randn(*param.data.size()) * np.sqrt(levy_stable.rvs(alpha=args.iterate_noise_alpha/2, beta=1, scale=np.cos(np.pi*args.iterate_noise_alpha/4)**(2/args.iterate_noise_alpha), size=(param.data.size()[:2])))[:, :, np.newaxis, np.newaxis]
                    else:
                        noise = levy_stable.rvs(alpha=args.iterate_noise_alpha, beta=0, scale=1, size=(param.data.size()[:2]))[:, :, np.newaxis, np.newaxis]
                    param.data.add_(torch.from_numpy(iterate_noise_scale * noise).cuda()) if args.use_cuda else param.data.add_(torch.from_numpy(noise))
                if len(param.data.size()) == 2:
                    if param.data.size()[0] == num_classes:
                        continue
                    if args.iterate_noise == 2:  
                        noise = levy_stable.rvs(alpha=args.iterate_noise_alpha, beta=0, scale=1, size=param.data.size())
                    elif args.iterate_noise == 3:
                        noise = np.random.randn(*param.data.size()) * np.sqrt(levy_stable.rvs(alpha=args.iterate_noise_alpha/2, beta=1, scale=np.cos(np.pi*args.iterate_noise_alpha/4)**(2/args.iterate_noise_alpha), size=(param.data.size()[0])))[:, np.newaxis]
                    else:
                        if args.iterate_noise_alpha==2.0:
                            noise = np.random.randn(param.data.size()[0])[:, np.newaxis]
                        else:
                            noise = levy_stable.rvs(alpha=args.iterate_noise_alpha, beta=0, scale=1, size=(param.data.size()[0]))[:, np.newaxis]
                    param.data.add_(torch.from_numpy(iterate_noise_scale * noise).cuda()) if args.use_cuda else param.data.add_(torch.from_numpy(noise))
                # HACK: Applies both to bias and batch_norm parameters
                if len(param.data.size()) == 1:
                    if args.iterate_noise_bias:
                        raise NotImplementedError

        if STOP:
            print("\n## Final evaluation: ")
            print('train eval')
            tr_hist, tr_outputs = eval(train_loader_eval, net, crit, args)
            print('test eval')
            te_hist, te_outputs = eval(test_loader_eval, net, crit, args)

            evaluation_history_TRAIN.append([i + 1, *tr_hist])
            evaluation_history_TEST.append([i + 1, *te_hist])

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            else:
                print('Folder already exists, beware of overriding old data!')

            # save the setup
            torch.save(args, args.save_dir + '/args.info')
            # save the outputs
            torch.save(te_outputs, args.save_dir + '/te_outputs.pyT')
            torch.save(tr_outputs, args.save_dir + '/tr_outputs.pyT')
            # if exists, save the script
            if script:
                write_message_to_file(script, args.save_dir + '/script.sh')

            # save the model
            torch.save(net, args.save_dir + '/net.pyT')

            # save the logs
            torch.save(training_history, args.save_dir + '/training_history.hist')
            torch.save(evaluation_history_TEST, args.save_dir + '/evaluation_history_TEST.hist')
            torch.save(evaluation_history_TRAIN, args.save_dir + '/evaluation_history_TRAIN.hist')

            end_time = time.time()
            total_time = end_time - begin_time
            time_secs = str(datetime.timedelta(seconds=total_time))

            print("\nTotal Time: " + time_secs)
            break