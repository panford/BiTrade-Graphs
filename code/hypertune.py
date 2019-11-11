from run_classifier import do_train 
import argparse, os
import torch, numpy as np
from utils import getdata
def init_params(args = None):
    params = dict()
    lr_list = [0.001, 0.01, 0.1]
    weight_decay_list = [0.0005, 0.005, 0.05]
    k = [1,2,3,4,5]
    param_defaults = dict(lr = lr_list, weight_decay = weight_decay_list)
    if args:
        for a in args:
            if a in param_defaults.keys():
                params[a] = param_defaults[a]
        return params
    
    else:
        return param_defaults

def tuner(hparams, tparams): #tparams are the parameters to be tuned and hparams are all other parameters that need no tuning
    index = 0
    results = []
    results_lookup = dict()
    hparams.update({t:tparams[t][0] for t in tparams.keys()})
    for t in tparams.keys():
        
        print("t: ",t)
        for j in tparams[t]:
            print("j: ", j)
            print('weight default: ', hparams['weight_decay'])
            hparams[t] = j
            test_acc = do_train(**hparams)
            
            results.append(test_acc)
            results_lookup[index] = hparams
            index += 1
    print(results)
    best_acc_ind = results.index(max(results))
    best_params = results_lookup[best_acc_ind]
    return max(results), best_params

def print_params(best_params):
    print(best_params)
    return

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = 'all', type = str, help = "Enter model name to run or enter 'all' to run all models in chain")
    parser.add_argument("--data_dir", type = str,  default = '../data/processed')
    parser.add_argument("--output_dir", type = str, default = '../tuned_results') 
    parser.add_argument("--input_filename", type = str, default = 'trade_savez_files.npz')
    parser.add_argument("-tune", type = list, default = [] , help = "List all paramaters to tune with defaults")
#     parser.add_argument("-with_defaults", type = int, default = 10)
    parser.add_argument("-epochs", type = int, default = 10)
    parser.add_argument("-lr_list", type = list, default = [] , help = "Specify range of learning rates in list for tuning")
    parser.add_argument("-weight_decay_list", type = list, default = [], help = "Specify range of weight decay in list for tuning")
    
    
    args = parser.parse_args()
    input_file_path = os.path.join(args.data_dir, args.input_filename)
    input_data = getdata(input_file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running classifiers")
    
    hparams = dict(model_name = args.model_name, 
                        data_dir = args.data_dir, 
                        output_dir = args.output_dir, 
                        input_filename = args.input_filename,
                        epochs = args.epochs,
                        lr = 0.001,
                        weight_decay = 5e-3
                       )
    tparams = dict()
    if args.tune:
        tparams = init_params(args.tune)        
    else:
        tparams = init_params()
        
    if args.lr_list:
        tparams['lr'] = args.lr_list
        
    if args.weight_decay_list:
        tparams['weight_decay'] = args.weight_decay_list
        
    best_acc, best_params = tuner(hparams, tparams)
# python hypertune.py --model_name 'all' -tune ['lr', 'weight_decay'] -lr [0.0001, 0.001, 0.01] -weight_decay_list [0.0005, 0.005, 0.05]


if __name__ == "__main__":
    main()