import torch, pandas as pd, numpy as np, math, scipy, heapq#, zarr; 
import pickle, logging, os
from PIL import Image
import torch.nn as nn
import json
from datetime import datetime

from numpy.linalg import solve;
#from spgl1 import spg_bp
from scipy.linalg import svd, norm, inv, lstsq;
from scipy.stats import multivariate_normal as mvn;
from scipy.stats import levy_stable, percentileofscore
from scipy.special import logsumexp;
from scipy.optimize import nnls
from torchvision import datasets, transforms; from datetime import datetime 
from torchvision.datasets.vision import VisionDataset
from torchvision import datasets, transforms;
import pickle, logging, os, json, matplotlib.pyplot as plt, matplotlib

EPS = {"Linf": 8/255, "L2": .5} #for AR
STATS = {"cifar10":{'mean': [0.491, 0.482, 0.447],'std': [0.247, 0.243, 0.262]}, "cifar100":{'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]}, "mnist": {'mean': [0.1307],'std': [0.3081]}} #for AR

AVAILABLE_PARAMS = (2, 4)

device = torch.device('cuda')
logger = logging.getLogger(); logger.setLevel(logging.CRITICAL)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

def get_col_prune_idx(X, max_compression_error):
  col_norms = np.linalg.norm(X, axis=0)
  sorted_X = X[:, np.argsort(col_norms)]
  idx_cutoff = np.argmax(np.sqrt(np.cumsum((sorted_X ** 2).sum(0)))/np.linalg.norm(X) > max_compression_error)
  return np.argsort(col_norms)[:idx_cutoff]

### RESULTS FOLDER PROCESSING
def k_most_recently_modified(folders, k):
    # Help from ChatGPT
    modification_times = [(os.path.getmtime(folder), folder) for folder in folders]
    most_recent = heapq.nlargest(k, modification_times)
    return [folder for _, folder in most_recent]

class ResultsFolder(object):
    def __init__(self, results_root_folder, k_most_recent=0):
        self.results_root_folder = results_root_folder
        self.timestamp, self.folders, self.results_summary_folder = prepare_result_analysis(results_root_folder)
        if k_most_recent > 0:
            self.folders = k_most_recently_modified(self.folders, k_most_recent)
        # HACK: This self.args should be removed to include experiments with different datasets in a single folder.
        model, self.args, self.model_name, self.dataset_name, _, self.no_layers = get_model_info(self.folders[0], dream_team=True, x_type="x_final")
        if self.args.criterion == 'NLL':
            self.crit = torch.nn.CrossEntropyLoss(reduction='mean').to(self.args.device)    
        self.param_sizes = []
        with torch.no_grad():
            for pi, p in enumerate(model.parameters()):
                self.param_sizes.append(p.shape)
        self.total_params = [np.prod(size) for size in self.param_sizes]

def prepare_result_analysis(results_root_folder):
    assert (results_root_folder[-1] == "/")
    timestamp = results_root_folder.split("/")[1].split("_")[0];
    timestamp = timestamp if timestamp.isnumeric() else "00000000"
    folders = sorted([r for r in [results_root_folder + f + "/" for f in os.listdir(results_root_folder)] if os.path.isdir(r) and os.path.exists(r + "net.pyT")])
    results_summary_folder = results_root_folder + f"{timestamp}_results_summary/"
    os.makedirs(results_summary_folder) if not os.path.exists(results_summary_folder) else None
    return timestamp, folders, results_summary_folder

def get_model_info(m, dream_team, x_type):
    model_file_name = "avg_net.pyT" if x_type == "x_mc" else "net.pyT"
    model = torch.load(m + model_file_name,map_location='cpu')
    args = torch.load(m + "args.info",map_location='cpu')
    model_name = args.model.upper()
    dataset_name = args.dataset.upper()
    layers = get_layers(model, dream_team=dream_team)
    no_layers = len(layers)
    return model, args, model_name, dataset_name, layers, no_layers

def get_model_info_files(model_file_path, args_file_path, dream_team):
    model = torch.load(model_file_path,map_location='cpu')
    args = torch.load(args_file_path,map_location='cpu')
    model_name = args.model.upper()
    dataset_name = args.dataset.upper()
    layers = get_layers(model, dream_team=dream_team)
    no_layers = len(layers)
    return model, args, model_name, dataset_name, layers, no_layers

def simulate_mv_stable(d , n, alp, elliptic=True):
    if(elliptic):
        phi = levy_stable.rvs(alp/2, 1, loc=0, scale=2*np.cos(np.pi*alp/4)**(2/alp), size=(1,n))
        mv_stable = np.sqrt(phi) * np.random.randn(d,int(n))
    else:
        mv_stable = levy_stable.rvs(alpha = alp, beta = 0, scale = 1, size=(d,n))
    
    return mv_stable

def find_m(N, m=0):
    if m == 0:
        m = int(np.sqrt(N)); assert N < 3e7+1
    while N % m != 0:
       m -= 1
    return m

### MISC
    
def get_layers(model, dream_team):
    layers = [layer for layer in model.parameters() if (layer.dim() >= 2)]
    if dream_team:
        layers[-1] = layers[-1].T
    return layers

def get_pruned_layers(pruned_layers_str):
    assert "10" not in pruned_layers_str
    return [int(i) for i in pruned_layers_str]

def get_total_params(model):
    return sum([np.prod(layer.shape) for layer in model.parameters()])

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        ret = pickle.load(f)
    return ret 

def clean_folder(folder):
    #return folder.replace("results/","").replace("/", "")
    return folder.split("/")[-2]

def get_layer_to_numpy(p):
    with torch.no_grad():
        a = p.to("cpu").numpy()
    return a

def get_folder_idx(orig_folder_length, k, T):
    no_exps = orig_folder_length // T
    start = (k-1) * no_exps
    end = k  * no_exps if k < T else orig_folder_length
    return no_exps, start, end

def get_results_part_string(eparallel, no_samples):
    results_part_string = "" if eparallel == "_1_1" else "_part" + eparallel
    if results_part_string == "":
        if no_samples != -1:
            results_part_string = f"_{no_samples}_samples"
    return results_part_string
def sample_result_folders(results_folder, no_samples):
    df = pd.read_csv(results_folder.results_summary_folder + results_folder.timestamp + "_x_mc_results.csv")
    df = df.sort_values("lr_b")
    chosen_folders = df.folder.iloc[list(map(int, np.linspace(0, len(df.folder)-1, no_samples)))].to_list()
    return [f for f in results_folder.folders if any([k + "/" in f for k in chosen_folders])]

class ECG5000(VisionDataset):
    """New dataset."""

    def __init__(self, root, transform=None, target_transform=None, train=True):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = np.load(root + f"/ECG5000/X_{'train' if train else 'test'}.npy")
        self.targets = np.load(root + f"/ECG5000/y_{'train' if train else 'test'}.npy")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

### DATA

class ToLong:
    def __call__(self, tensor):
        return tensor.long()
    
class ToTensor:
    def __call__(self, array):
        return torch.tensor(array)


def get_data(args):

    if args.dataset == 'cifar100':
        data_class = 'CIFAR100'
        num_classes = 100
        input_dim = 32 * 32 * 3
        stats = {
            'mean': [0.5071, 0.4867, 0.4408] ,
            'std': [0.2675, 0.2565, 0.2761]
            }
    elif 'cifar10' in args.dataset:
        data_class = 'CIFAR10'
        num_classes = 10
        input_dim = 32 * 32 * 3
        stats = {
                'mean': [0.491, 0.482, 0.447],
                'std': [0.247, 0.243, 0.262]
                }
    
    elif args.dataset in ["mnist"]:
        data_class = 'MNIST'
        num_classes = 10
        input_dim = 28 * 28   
        stats = {'mean': [0.1307], 'std': [0.3081]}
    elif 'ecg5000' in args.dataset:
        data_class = 'ecg5000'
        ecg5000_data_loader = ECG5000
        num_classes = 2
        input_dim = 140
        stats = {
            'mean': [0.],
            'std': [1.]
            }
    else:
        raise ValueError("unknown dataset")
    try:
        args.data_scale
    except AttributeError:
        setattr(args, 'data_scale', 1)
    
    # input transformation w/o preprocessing for now
    stats["std"] = [s/args.data_scale for s in stats["std"]]
    
    trans = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**stats)
        ]

    if args.dataset == "ecg5000":
        tr_data = ecg5000_data_loader(
            root=args.path,
            train=True,
            transform=transforms.Compose([ToTensor()] + trans[1:2]),
            target_transform=transforms.Compose([ToTensor(), ToLong()]),
        )
        te_data = ecg5000_data_loader(
            root=args.path,
            train=False,
            transform=transforms.Compose([ToTensor()] + trans[1:2]),
            target_transform=transforms.Compose([ToTensor(), ToLong()]),
        )
    else:

        tr_data = getattr(datasets, data_class)(
            root=args.path,
            train=True,
            download=True,
            transform=transforms.Compose(trans)
            )

        te_data = getattr(datasets, data_class)(
            root=args.path,
            train=False,
            download=True,
            transform=transforms.Compose(trans)
            )

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_train,
        shuffle=True,
        )

    train_loader_eval = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_eval,
        shuffle=True,
        )

    test_loader_eval = torch.utils.data.DataLoader(
        dataset=te_data,
        batch_size=args.batch_size_eval,
        shuffle=True,
        )
    images, labels = next(iter(test_loader_eval))
      
    return train_loader, test_loader_eval, train_loader_eval, num_classes, input_dim


### EVALUATION

def get_loss_and_accuracy(df, fi, folder):
    df.loc[fi, "training_loss"] = torch.load(folder + "evaluation_history_TRAIN.hist",map_location='cpu')[-1][1]
    df.loc[fi, "test_loss"] = torch.load(folder + "evaluation_history_TEST.hist",map_location='cpu')[-1][1]
    df.loc[fi, "loss_diff"] = np.abs(df.loc[fi, "test_loss"] - df.loc[fi, "training_loss"])
    df.loc[fi, "training_accuracy"] = torch.load(folder + "evaluation_history_TRAIN.hist",map_location='cpu')[-1][2]
    df.loc[fi, "test_accuracy"] =     torch.load(folder + "evaluation_history_TEST.hist",map_location='cpu')[-1][2]
    df.loc[fi, "accuracy_diff"] = np.abs(df.loc[fi, "test_accuracy"] - df.loc[fi, "training_accuracy"])
    df.loc[fi, "training_loss_avg"] = torch.load(folder + "evaluation_history_AVGTRAIN.hist",map_location='cpu')[-1][1]
    df.loc[fi, "test_loss_avg"] = torch.load(folder + "evaluation_history_AVG.hist",map_location='cpu')[-1][1]
    df.loc[fi, "loss_diff_avg"] = np.abs(df.loc[fi, "test_loss_avg"] - df.loc[fi, "training_loss_avg"])
    df.loc[fi, "training_accuracy_avg"] = torch.load(folder + "evaluation_history_AVGTRAIN.hist",map_location='cpu')[-1][2]
    df.loc[fi, "test_accuracy_avg"] =     torch.load(folder + "evaluation_history_AVG.hist",map_location='cpu')[-1][2]
    df.loc[fi, "accuracy_diff_avg"] = np.abs(df.loc[fi, "test_accuracy_avg"] - df.loc[fi, "training_accuracy_avg"])
    
def accuracy(out, y):
    _, pred = out.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float() / y.size(0)

def eval(eval_loader, net, crit, args, if_print=True):

    net.eval()
    # run over both test and train set
    total_size = 0
    total_loss = 0
    total_acc = 0
    outputs = []

    with torch.no_grad():
        # P = 0  # num samples / batch size
        for x, y in eval_loader:
            # P += 1
            # loop over dataset
            x, y = x.to(args.device), y.to(args.device)
            out = net(x)
            outputs.append(out)
            loss = crit(out, y)
            prec = accuracy(out, y)
            bs = x.size(0)

            total_size += int(bs)
            total_loss += float(loss) * bs
            total_acc += float(prec) * bs

        hist = [total_loss / total_size, total_acc / total_size]
        if if_print:
            print(hist)

        return hist, outputs
    
### TRAINING NEURAL NETWORKS

def get_convergence_criteria(dataset, method, loss_crit, accuracy_crit):
    """Determining the convergence criteria according to determination method and/or dataset."""
    if method == "dataset":
        if dataset == "cifar100":
            loss_crit = 1e-2
            accuracy_crit = .99
        elif "dcase" in dataset:
            loss_crit = np.inf
            accuracy_crit = 95.
        elif "esc" in dataset:
            loss_crit = np.inf
            accuracy_crit = 100.
        else:
            loss_crit = 5e-5
            accuracy_crit = 100.
    if method == "none":
        loss_crit = -np.inf
        accuracy_crit = np.inf
    return loss_crit, accuracy_crit

def get_accuracy_lower_bound(num_classes):
    return 1.25*(100/num_classes) if num_classes > 2 else 53

def write_message_to_file(msg, file_path):
    f = open(file_path, 'a')
    f.write(msg + "\n")
    f.close()

def print_experiment_script(script, file_traj):
    msg = "*** EXPERIMENT SCRIPT START ***\n" + script + "\n*** EXPERIMENT SCRIPT END ***\n"
    if script:
        write_message_to_file(msg, file_traj)
        print(msg)

def determine_iter_record_freq(i, record_freq):
    if record_freq >= 0:
        return record_freq
    if record_freq < 0:
        if i < 100:
            return 10
        if i < 500:
            return 100
        elif i < 5000:
            return 500
        elif i < 10000:
            return 1000
        elif i < 50000:
            return 5000
        else:
            return 25000
        
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def get_col_prune_idx(X, pruning_ratio, pruning_criterion, compression_error_metric):
  # np.save("./X_example.npy", X)
  col_norms = np.linalg.norm(X, axis=0)
  if pruning_criterion == "mce":
    sorted_X = X[:, np.argsort(col_norms)]
    if compression_error_metric == "fro_squared":
      idx_cutoff = np.argmax(np.cumsum((sorted_X ** 2).sum(0))/(np.linalg.norm(X)**2) > pruning_ratio)
    else:
      idx_cutoff = np.argmax(np.sqrt(np.cumsum((sorted_X ** 2).sum(0)))/np.linalg.norm(X) > pruning_ratio)
  elif pruning_criterion == "parameter_ratio":
    idx_cutoff = int(len(col_norms) * pruning_ratio)
  return np.argsort(col_norms)[:idx_cutoff]

def unprunable_layer(p, num_classes, first_conv):
  if (len(p.shape) == 1):
    return True
  if (len(p.shape) == 4) and first_conv:
    print("Not pruning the first convolution layer.")
    first_conv = False
    return True
  if (len(p.shape) == 2) and (num_classes in p.shape):
    print("Not pruning the final linear layer.")
    return True
  else:
    return False
  
def backwards_compatibility(args):
  if not (hasattr(args, "convergence_method") or  hasattr(args, "iterate_noise") or hasattr(args, "iterate_noise_alpha") or  hasattr(args, "iterate_noise_scale")):
    args.convergence_method, args.iterate_noise, args.iterate_noise_alpha, args.iterate_noise_scale = "dataset", 0, 1.0, 0.0
  return args

def col_pruning(folder, criterion_ratio, evaluate=True, pruning_type="kernel", pruning_criterion="mce", global_pruning=False, compression_error_metric="fro_squared", ignore_last_layer=False, device=None):
    model, args, model_name, dataset_name, layers, no_layers = get_model_info(folder, dream_team=True, x_type="x_final")
    args.device = device
    model = model.to(device)
    train_loader, test_loader_eval, train_loader_eval, num_classes, input_dim = get_data(args)
    # HACK
    crit = nn.CrossEntropyLoss(reduction='mean')
    try:
      model.adaptive_ff_layers
    except AttributeError:
      model.adaptive_ff_layers = False
    total_unpruned_parameters = 0
    first_conv = True
    if criterion_ratio > 0:
      if global_pruning:
        pass
      else:
        with torch.no_grad():
            for pi, p in enumerate(layers): 
                print(num_classes)
                if unprunable_layer(p, num_classes, first_conv):
                  first_conv = False if (len(p.shape) == 4) and first_conv else first_conv 
                  if (p.dim() == 2) and (num_classes in p.shape):
                    last_layer_params = np.prod(p.shape)
                  continue
                a = get_layer_to_numpy(p).copy(); assert len(a.shape) in AVAILABLE_PARAMS
                if len(p.shape) == 4:
                    a_dims = a.shape
                    if pruning_type == "filter":
                      a = np.reshape(a, (np.prod(a.shape[:1]), np.prod(a.shape[1:])))    
                    else:
                      a = np.reshape(a, (np.prod(a.shape[:2]), np.prod(a.shape[2:])))
                idx = get_col_prune_idx(a.T, criterion_ratio, pruning_criterion=pruning_criterion, compression_error_metric=compression_error_metric)
                print(f"Layer shape: {a.T.shape}. Pruning {np.round(len(idx)/a.shape[0], 4)} of the columns ({len(idx)} columns).")
                a[idx, :] = 0.
                total_unpruned_parameters += int(a.shape[0] - len(idx)) * a.shape[1]
                if len(p.shape) == 4:
                    a = np.reshape(a, (a_dims))
                p.data = torch.Tensor(a)  
      total_remaining_params = sum([np.prod(layer.shape) for layer in model.parameters() if layer.dim() not in AVAILABLE_PARAMS])
      total_unpruned_parameters += last_layer_params
      last_layer_params = last_layer_params if ignore_last_layer else 0 
      remaining_parameter_ratio = (total_unpruned_parameters + total_remaining_params - last_layer_params) / (get_total_params(model) - last_layer_params)
    else:
      print("No pruning.")
      remaining_parameter_ratio = 1.
    if evaluate:
      pruned_train_hist, _ = eval(train_loader_eval, model.to(device), crit, args, if_print=False)
      pruned_test_hist, _ = eval(test_loader_eval, model.to(device), crit, args, if_print=False)
    return pruned_train_hist[0], pruned_train_hist[1], pruned_test_hist[0], pruned_test_hist[1], 1 - remaining_parameter_ratio, args
