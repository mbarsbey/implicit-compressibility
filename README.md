# Readme File for _Implicit Compressibility of Overparametrized Neural Networks Trained with Heavy-Tailed SGD_

Here we describe the steps necessary to reproduce the main results presented in the paper. Please create an appropriate Python environment with the packages in the `requirements.txt` installed, and switch to it before you run the experiments.

## Datasets
The data files will be automatically be downloaded for the datasets MNIST, CIFAR10, and CIFAR100. If you want to run experiments with the ECG5000 dataset, please download it from http://www.timeseriesclassification.com/description.php?Dataset=ECG5000 and place it in the folder `data/ECG5000/` as four numpy files, `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`. These files should include the training and test input and output values respectively.

## Training the algorithms

The principal file for training the algorithms is `main.py`. An example command for starting a training run is:

`python main.py --iterate_noise 1 --iterate_noise_scale 0.003 --iterate_noise_alpha 1.75 --model fcn  --width 5000 --depth 2 --dataset mnist --lr 0.01 --batch_size_train 5000 --save_dir results/mnist_experiment --seed 1 --eval_freq 100 --convergence_accuracy 95`

Here, the flags govern the following training features:

- `--iterate_noise`: Iterate noise type. While 0 implies no noise added during training, 1, 2, and 3 correspond to the noise types introduced in the paper.
- `--iterate_noise_scale`: Scale of the noise vectors, corresponds to $\sigma$ from the main paper.
- `--iterate_noise_alpha`: Tail index of the sampled noise vectors, corresponds to $\alpha$ from the main paper.
- `--model`: Model type. Choose `fcn` for fully connected networks, `vgg11_wide` for CIFAR10 experiments with CNN models. 
- `--width`: Width of the model, corresponds to $n$ from the main paper. Only applies to FCN models.
- `--depth`: Depth of the model. Only applies to FCN models. If a model with `d` hidden layers is desired, `d+1` should be entered for this parameter.
- `--dataset`: Dataset to be trained on. Choices for the paper are `ecg5000`, `mnist`, `cifar10`, `cifar100`.
- `--lr`: Learning rate.
- `--batch_size_train`: Training batch size.
- `--save_dir`: The directory to which the results will be saved.
- `--seed`: Seed for RNG. The experiments in the paper use 1, 2, 3, 4, 5 for replicated experiments, 1 for experiments run a single time.
- `--eval_freq`: The iteration frequency with which the model will be evaluated on train and test sets. The experiments in the paper use 10 for ECG5000, and 100 for the remaining experiments.
- `--convergence_accuracy`: The accuracy at which the experiment will stop.
- `--iterate_noise_filter`: When provided, this flag results in noise variables to be sampled for each filter separately (as opposed to each kernel separately). Only applies to CNN models. Is used for CIFAR10 CNN experiments in the main paper.

## Pruning experiments

After desired experiments are run, collect the completed experiment folders you want to conduct pruning and analyze, in a single folder, e.g. `results/mnist_experiments`. Then, use `analyze.py` to proceed, an example use is provided below:  

`python analyze.py --results_folder results/mnist_experiments  --pruning_criterion mce --ignore_last_layer --pruning_type filter`

Here, the flags govern the following training properties:

- `--results_folder`: The folder in which the completed experiment folders are located.
- `--pruning_criterion`: The criterion according to which the pruning will be completed. `mce` will lead to pruning columns until (squared) Frobenius norm of the error over that of the original parameter matrix equals 0.1 (see paper for details). `parameter_ratio` will conduct pruning for 0.0, 0.1, ... 0.9, 1.0 of columns in each layer.
- `--ignore_last_layer`: Excluding the last layer from pruning ratio computations. Is true for the results in the paper.
- `--pruning_type`: Determines how the convolution parameters are matricized. Only applies to CNN model. When provided `filter`, each filter becomes a column, `kernel` leads to each kernel becoming a column. Set to `filter` in the main paper experiments.

When the analysis finishes, the results are saved in the form of a `csv` file in a subfolder in the experiments folder.

## Hyperparameter

See Appendix E of the paper for the hyperparameters of the experiments presented in the main paper.