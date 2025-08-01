import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import json
from pytorch_lightning.loggers import CSVLogger
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def load_config(config_file):
    """Load configuration from file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def get_model_config(config, model_name, num_items):
    """Get model-specific configuration"""
    common_params = config['common_params']
    model_config = config['model_configs'][model_name]
    
    # Check if the provided grid size is a default one
    if str(num_items) in map(str, model_config['hyperparameters'].keys()):
        # Combine common params with model-specific params for this grid size
        full_config = {
            **common_params,
            **model_config['hyperparameters'][str(num_items)],
            'model_name': model_name,
            'num_items': num_items
        }
    else:
        # Use default hyperparameters
        full_config = {
            **common_params,
            'model_name': model_name,
            'num_items': num_items,
            'lr': 0.005,       # Default value, will be overridden if specified
            'sigma': 0.1,    # Default value, will be overridden if specified
            'n_samples': 1  # Default value, will be overridden if specified
        }
    return full_config



parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="path to config file", default="configs/knapsack_DYSconfig.json")
parser.add_argument("--model_name", type=str, required=True, help="name of models")
parser.add_argument("--num_items", type=int, help="Number of Items")
parser.add_argument("--dim", type=int, help="Number of dimesnion of the weight space")
parser.add_argument("--capacity", type=int, help="Capacity")

parser.add_argument("--seed", type=int, help="random seed")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--max_epochs", type=int, help="maximum epochs")
parser.add_argument("--num_data", type=int, help="Number of Training Data")
parser.add_argument("--num_feat", type=int, help="Number of Features")
parser.add_argument("--deg", type=int, help="degree of misspecifaction")
parser.add_argument("--noise_width", type=float, help="noise width misspecifaction")
parser.add_argument("--normalize", action='store_true', help="Heuristics to be used?")
parser.add_argument("--scheduler", action="store_true", help="Set this flag to enable it")
parser.add_argument("--lr", type=float, help="learning rate")

parser.add_argument("--tau", type=float, help="Tau, the parameter of quadratic regularizer")
parser.add_argument("--alpha", type=float, help="the parameter of DYS")
parser.add_argument("--decay_param", type=float, help="the parameter of DYS DECAY")
parser.add_argument("--numiter", type=int, help="Number of DYS Iterations")


args = parser.parse_args()
# Load and get model-specific configuration
config = load_config(args.config)
config = get_model_config(config, args.model_name, args.num_items)

# Update config with command line arguments (only if they are provided)
for key, value in vars(args).items():
    if key not in ['config', 'model_name', 'num_items'] and value is not None:
        config[key] = value

# Set random seed
torch.manual_seed(config['seed'])
num_data, num_feat = config['num_data'], config['num_feat']
num_items, dim, capacity =  config['num_items'],  config['dim'] , config ['capacity']

caps = [int(num_items * capacity)] * dim # capacity
deg, noise_width = config['deg'], config['noise_width']
modelname = config['model_name']
normalize = config['normalize']
relax = config['relax']


batch_size, max_epochs = config['batch_size'], config['max_epochs']
lr, scheduler, seed = config['lr'], config['scheduler'], config['seed']

weights, feats, costs = pyepo.data.knapsack.genData(num_data+1000, num_feat,
                     num_items, deg=deg, dim=dim, noise_width=noise_width, seed= seed)

from sklearn.model_selection import train_test_split
x_train, x_test, c_train, c_test = train_test_split(feats, costs, 
                                                    test_size=1000, 
                                                    random_state=seed)
x_train, x_val, c_train, c_val  = train_test_split(x_train, c_train,
                                                    test_size=0.2,
                                                     random_state=seed) 
# from KnapsackSolver import knapsackModel
# optmodel = knapsackModel(weights, caps, relax= True)
optmodel = pyepo.model.grb.knapsackModel( weights, caps) 
if relax:
    optmodel_train = pyepo.model.grb.knapsackModelRel( weights, caps) 
else:
    optmodel_train = optmodel


# get optDataset
dataset_train = pyepo.data.dataset.optDataset(optmodel_train, x_train, c_train)
dataset_val = pyepo.data.dataset.optDataset(optmodel, x_val, c_val)
dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=19)
loader_val  = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=19)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=19)
from torch import nn

# build linear model

import pytorch_lightning as pl
from LightningDFL_Models import  DYS
from MLmodels import LinearRegression
log_dir = os.getcwd() + "/ResultECAI/KnapsackResults/"

reg = LinearRegression(num_feat, num_items) # init model


if 'DYS' in modelname:

    dflloss = config['dflloss']
    tau =  config['tau']
    alpha = config['alpha']
    decay =  config['decay_param']
    numiter = config['numiter']
    logger =  CSVLogger (log_dir, name='dys{}_normalize{}_deg{}_noise{}_numitems{}'.format(dflloss, 
            normalize, deg, noise_width , num_items) )
    model = DYS(net= reg, dflloss = dflloss, tau=tau, alpha = alpha,decay_param = decay, num_iter=numiter,
                lr=lr, scheduler=scheduler, seed=seed, optmodel= optmodel, max_epochs=max_epochs)


trainer = pl.Trainer(max_epochs=max_epochs, check_val_every_n_epoch=max_epochs, logger=logger)
trainer.validate(model, dataloaders = loader_val )
trainer.fit(model,  train_dataloaders= loader_train, val_dataloaders= loader_val)
print("Test Result: ", trainer.test(dataloaders = loader_test) )


