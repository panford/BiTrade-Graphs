import argparse
import os
from torch_geometric.nn import GAE, VGAE
from utils import getdata, run_VGAE, run_GAE

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, default="all")
    parser.add_argument("--data_dir", type=str, default= "../data/processed")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("-input_filename", type=str, default="trade_savez_files.npz")
    parser.add_argument("-lr", type = float, default=0.001, help= "Enter learning rate for training")
    parser.add_argument("-weight_decay", type=float, default=0.005, help="Enter weight decay rate")
    parser.add_argument("-epochs", default=10, help = "Input number of training epochs")
    args = parser.parse_args()
    input_file_path = os.path.join(args.data_dir, args.input_filename)
    input_data = getdata(input_file_path)
    
    print("Running link prediction...")

    if args.model_name == 'all':
        run_GAE(input_data, args.output_dir, epochs = args.epochs, lr = args.lr, weight_decay = args.weight_decay)
        run_VGAE(input_data, args.output_dir, epochs = args.epochs, lr = args.lr, weight_decay = args.weight_decay)

    elif args.model_name == 'GAE':
        run_GAE(input_file_path, args.output_dir, epochs = args.epochs, lr = args.lr, weight_decay = args.weight_decay)
        
    elif args.model_name == 'VGAE':
        run_VGAE(input_data, args.output_dir, epochs = args.epochs, lr = args.lr, weight_decay = args.weight_decay)

        
if __name__ == "__main__":
    main()