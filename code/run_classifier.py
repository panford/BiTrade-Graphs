import os
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from models import GCNet, GATNet, AGNNet, GAEncoder, VGAEncoder 
from utils import classifier_train_test, getdata
from hypertune import tuner, print_params

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = 'all', type = str, help = "Enter model name to run or enter 'all' to run all models in chain")
    parser.add_argument("--data_dir", type = str,  default = '../data/processed')
    parser.add_argument("--output_dir", type = str, default = '../results') 
    parser.add_argument("--input_filename", type = str, default = 'trade_savez_files.npz')
    parser.add_argument("-epochs", type = int, default = 5)
    parser.add_argument("-lr", type = float, default = 0.001, help = "Enter learning rate for training")
    parser.add_argument("-weight_decay", type = str, default = 5e-3, help = 'Enter weight decay')
    parser.add_argument("-hparam_tune", type = bool, default = 'n', help = "Enter 'y' or '1' for hyperparameter tuning or leave default 'no'")
    

    args = parser.parse_args()
    input_file_path = os.path.join(args.data_dir, args.input_filename)
    input_data = getdata(input_file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running classifiers")

    if args.model_name == "all":
        classifier_train_test(GCNet, input_data, args.output_dir, args.epochs,args.lr, args.weight_decay)
        classifier_train_test(GATNet, input_data, args.output_dir, args.epochs, args.lr, args.weight_decay)
        classifier_train_test(AGNNet, input_data, args.output_dir, args.epochs, args.lr, args.weight_decay)
    
    elif args.model_name == "GCNet":
        classifier_train_test(GCNet, input_data, args.output_dir, args.epochs, args.lr, args.weight_decay)
    
    elif args.model_name == "GATNet":
        classifier_train_test(GATNet, input_data, args.output_dir, args.epochs, args.lr, args.weight_decay)
   
    elif args.model_name == "AGNNet":
        classifier_train_test(AGNNet, input_data, args.output_dir, args.epochs, args.lr, args.weight_decay)
    
    else:
        print("sorry! model not implemented")
        
    if args.hparam_tune == 1 or 'y':
        best_params = tuner(params)
        print_params(best_params)

        
if __name__  == "__main__":
    main()