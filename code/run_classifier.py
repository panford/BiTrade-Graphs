import os
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from models import GCNet, GATNet, AGNNet, GAEncoder, VGAEncoder 
from utils import classifier_train_test, getdata

def do_train(model_name, data_dir, output_dir, input_filename, epochs, lr, weight_decay):
    input_file_path = os.path.join(data_dir, input_filename)
    input_data = getdata(input_file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running classifiers")

    if model_name == "all":
        classifier_train_test(GCNet, input_data, output_dir, epochs, lr, weight_decay)
        classifier_train_test(GATNet, input_data, output_dir, epochs, lr, weight_decay)
        classifier_train_test(AGNNet, input_data, output_dir, epochs, lr, weight_decay)
    
    elif model_name == "GCNet":
        acc = classifier_train_test(GCNet, input_data, output_dir, epochs, lr, weight_decay)
    
    elif model_name == "GATNet":
        acc = classifier_train_test(GATNet, input_data, output_dir, epochs, lr, weight_decay)
   
    elif model_name == "AGNNet":
        acc = classifier_train_test(AGNNet, input_data, output_dir, epochs, lr, weight_decay)
    
    else:
        print("sorry! model not implemented")
        
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = 'all', type = str, help = "Enter model name to run or enter 'all' to run all models in chain")
    parser.add_argument("--data_dir", type = str,  default = '../data/processed')
    parser.add_argument("--output_dir", type = str, default = '../results') 
    parser.add_argument("--input_filename", type = str, default = 'trade_savez_files.npz')
    parser.add_argument("-epochs", type = int, default = 5)
    parser.add_argument("-lr", type = float, default = 0.001, help = "Enter learning rate for training")
    parser.add_argument("-weight_decay", type = str, default = 5e-3, help = 'Enter weight decay')
    

    args = parser.parse_args()
    
    input_params = dict(model_name = args.model_name, 
                        data_dir = args.data_dir, 
                        output_dir = args.output_dir, 
                        input_filename = args.input_filename, 
                        epochs = args.epochs, 
                        lr = args.lr, 
                        weight_decay = args.weight_decay)
    
    acc = do_train(**input_params)
        
if __name__  == "__main__":
    main()