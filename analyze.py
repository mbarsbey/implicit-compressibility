import numpy as np
import torch.nn as nn
import torch
import os
from exp_utils import get_layer_to_numpy, get_model_info, get_data, eval, col_pruning, get_total_params, get_timestamp, ResultsFolder, load_json, backwards_compatibility
import pandas as pd
import argparse
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use CUDA
# print(DEVICE)
# raise(exception)
parser = argparse.ArgumentParser()
parser.add_argument('--pruning_type', '-t', type=str, default="kernel", choices=["kernel", "filter"])
parser.add_argument('--pruning_criterion', '-c', type=str, default="mce", choices=["mce", "parameter_ratio"])
parser.add_argument('--compression_error_metric', '-m', type=str, default="fro_squared", choices=["frob", "fro_squared"])
parser.add_argument('--ignore_last_layer', '-i', action='store_true', default=False, help="Ignoring last layer in the computations for ratio of parameters pruned.")
parser.add_argument('--drop_constant_columns', '-d', action='store_true', default=False)
parser.add_argument('--results_folder', '-r', type=str, default="esc10")
parser.add_argument('--k_most_recent', '-k', type=int, default=0)
parser.add_argument('--pruning_file', '-f', type=str, default="")
parser.add_argument('--prev_results_file', '-p', type=str, default="")

args_script = parser.parse_args()

print(f"Pruning type: {args_script.pruning_type}.")

results_folder = args_script.results_folder + "/" if args_script.results_folder[-1] != "/" else args_script.results_folder

if args_script.pruning_file:
  pruning_dict = load_json(args_script.pruning_file)  
  print("Pruning dict found.")
  exp_keys = pruning_dict.keys()
else:
  pruning_ratios = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.] if args_script.pruning_criterion == "parameter_ratio" else [0.0, 0.1]
  exp_keys = [folder.split("/")[-2] for folder in ResultsFolder(results_folder, args_script.k_most_recent).folders]
  pruning_dict = {exp_key: pruning_ratios for exp_key in exp_keys}

df = pd.DataFrame(index=range(sum([len(pruning_dict[exp_key]) for exp_key in exp_keys])), columns=["exp_key", "model", "dataset", "lr", "batch_size", "width", "depth", "seed", "iterations", "convergence_accuracy", "it_noise", "it_noise_alpha", "it_noise_scale", "criterion_ratio", "pruning_criterion", "train_loss", "train_acc", "test_loss", "test_acc", "pruning_ratio"])
df_prev = pd.read_csv(args_script.prev_results_file) if args_script.prev_results_file else pd.DataFrame(columns=["exp_key", "criterion_ratio"])

c = 0
for exp_key in exp_keys:
  folder = results_folder + exp_key + "/" 
  print(f"\n{exp_key}")
  for pruning_ratio in pruning_dict[exp_key]:
    print(f"** Criterion ratio: {pruning_ratio}. Pruning criterion: {args_script.pruning_criterion}.**" )
    if len(df_prev.query(f"exp_key == '{exp_key}' and criterion_ratio == {pruning_ratio}")) == 1:
        df.iloc[c, :] = df_prev.query(f"exp_key == '{exp_key}' and criterion_ratio == {pruning_ratio}").iloc[0]
        print("Obtained from previous analysis.")
        c+=1
        continue
        
    output = col_pruning(folder, pruning_ratio, evaluate=True, pruning_type=args_script.pruning_type, pruning_criterion=args_script.pruning_criterion, compression_error_metric=args_script.compression_error_metric, ignore_last_layer=args_script.ignore_last_layer, device=DEVICE)
    args = output[-1]
    args = backwards_compatibility(args)
    iterations = args.iterations if args.convergence_method == "none" else -1
    convergence_accuracy = args.convergence_accuracy if args.convergence_method == "custom" else -1
    df.iloc[c, :] = exp_key, args.model, args.dataset, args.lr, args.batch_size_train, args.width, args.depth, args.seed, iterations, convergence_accuracy, args.iterate_noise, args.iterate_noise_alpha, args.iterate_noise_scale, pruning_ratio, args_script.pruning_criterion, *output[:-1]
    print(f"Train accuracy {output[-5]} test accuracy {output[-3]} with {output[-2]} pruning ratio.")
    c+=1
df["it_noise_alpha"] = df["it_noise_alpha"].mask(df["it_noise"]==0, 0.0)
df["it_noise_scale"] = df["it_noise_scale"].mask(df["it_noise"]==0, 0.0)
df = df.sort_values(["model", "width", "lr", "it_noise", "it_noise_alpha", "it_noise_scale"])

results_summary_folder = results_folder + "00000000_results_summary/"
if not os.path.exists(results_summary_folder):
  os.makedirs(results_summary_folder)

if args_script.drop_constant_columns:
  print("Ignored the constant columns:", df.loc[:, df.nunique() == 1].columns)
  df = df.loc[:, df.nunique() != 1]

if args_script.pruning_file:
  df.to_csv(args_script.pruning_file.replace("_dicts.json", f"_noiseless_{args_script.pruning_type}_{args_script.pruning_criterion}.csv"), index=False)
else:
  df.to_csv(results_summary_folder + f"results_{args_script.pruning_type}_{args_script.pruning_criterion}_{get_timestamp()}.csv", index=False)