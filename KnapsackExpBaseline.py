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
parser.add_argument("--config", type=str, help="path to config file", default="configs/knapsack_config.json")
parser.add_argument("--model_name", type=str, required=True, help="name of models")
parser.add_argument("--num_items", type=int, help="Number of Items")
parser.add_argument("--dim", type=int, help="Number of dimesnion of the weight space")
parser.add_argument("--capacity", type=int, help="Capacity")

# Optional arguments to override config values
parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--sigma", type=float, help="sigma value")
parser.add_argument("--n_samples", type=int, help="number of samples")
parser.add_argument("--seed", type=int, help="random seed")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--max_epochs", type=int, help="maximum epochs")
parser.add_argument("--num_data", type=int, help="Number of Training Data")
parser.add_argument("--num_feat", type=int, help="Number of Features")
parser.add_argument("--deg", type=int, help="degree of misspecifaction")
parser.add_argument("--noise_width", type=float, help="noise width misspecifaction")
parser.add_argument("--normalize", action='store_true', help="Heuristics to be used?")
parser.add_argument("--lambda_val", type=float, help="lambda_value of DBB")
parser.add_argument("--scheduler", action="store_true", help="Set this flag to enable it")
parser.add_argument("--relax", action="store_true", help="Use Linear Relaxation instead of solving ILPS")

parser.add_argument("--solve_ratio", type=float, help="solve ratio for caching")
parser.add_argument("--tau", type=float, help="Tau, the parameter of quadratic regularizer")
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
print ("CAPS", caps)
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
dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
dataset_val = pyepo.data.dataset.optDataset(optmodel, x_val, c_val)
dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=19)
loader_val  = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=19)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=19)

from torch import nn

# build linear model

import pytorch_lightning as pl
from LightningDFL_Models import  SPO, PFL, CAVE, DBB, CVX, PFY, NCEMAP, NCEMAP_Linear
from MLmodels import LinearRegression
# log_dir = os.getcwd() +  "/KnapsackResults/"
log_dir = os.getcwd() + "/ResultECAI/KnapsackResults/"


reg = LinearRegression(num_feat, num_items) # init model
if modelname=='SPO':
    solve_ratio = config['solve_ratio']
    name = 'spo_normalize{}_deg{}_noise{}_numitems{}'.format(normalize, deg, noise_width , num_items)
    if relax:
        name = 'relax'+name
    logger = CSVLogger(log_dir, name= name )
    model = SPO(net= reg,lr=lr, scheduler=scheduler, seed=seed, 
                optmodel= optmodel, optmodel_train= optmodel_train ,
                normalize=normalize, max_epochs=max_epochs,
            solve_ratio=solve_ratio, dataset=dataset_train)

if modelname=='NCEMAP':
    solve_ratio = config['solve_ratio']
    
    name = 'ncemap_normalize{}_deg{}_noise{}_numitems{}'.format(normalize, deg, noise_width, num_items)
    if relax:
        name = 'relax'+name
    logger = CSVLogger(log_dir, name= name)
    model = NCEMAP_Linear(net= reg,lr=lr, scheduler=scheduler, seed=seed, 
                          optmodel= optmodel, optmodel_train= optmodel_train , 
                          normalize=normalize, max_epochs=max_epochs,
        solve_ratio=solve_ratio, dataset=dataset_train)


if modelname=='PFL':

    logger = CSVLogger(log_dir, name='PFL_deg{}_noise{}_numitems{}'.format( deg,  noise_width, num_items))
    model = PFL(net= reg, lr=lr, scheduler=scheduler, max_epochs=max_epochs, seed=seed, optmodel= optmodel)


if modelname=='PFY':
    n_samples  = config['n_samples']
    sigma  = config['sigma']
    name = 'pfy_deg{}_noise{}_numitems{}'.format(deg, noise_width, num_items)
    if relax:
        name = 'relax'+name
    logger = CSVLogger(log_dir, name=name)
    model = PFY(net= reg, optmodel= optmodel, optmodel_train= optmodel_train ,
                n_samples=n_samples, sigma=sigma, max_epochs=max_epochs, lr=lr, 
                scheduler=scheduler, seed=seed)

if modelname=='CAVE':
    from pyepo.data.dataset_util import collate_fn
    name = 'cave_deg{}_noise{}_numitems{}'.format(deg,   noise_width, num_items)
    
    
    if relax:
        name = 'relax'+name
    dataset_train = pyepo.data.dataset_util.optDatasetConstrs(optmodel_train, x_train, c_train)
    loader_train = DataLoader(dataset_train, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=19)
    logger = CSVLogger(log_dir, name=name )
    model = CAVE(net= reg, optmodel= optmodel, optmodel_train= optmodel_train ,
                  lr=lr, scheduler=scheduler,max_epochs=max_epochs, seed=seed)

if 'CVX' in modelname:
    dflloss = config['dflloss']
    tau =  config['tau']
    # if dflloss in ["SCE+", "regret"]:
    logger =  CSVLogger (log_dir, name='cvx{}_normalize{}_deg{}_noise{}_numitems{}'.format(dflloss, 
            normalize, deg, noise_width , num_items) )
    model = CVX(net= reg, dflloss = dflloss, tau=tau,max_epochs=max_epochs,
                lr=lr, scheduler=scheduler, seed=seed, optmodel= optmodel)

print (f"Training of {modelname} to be started")
trainer = pl.Trainer(max_epochs=max_epochs, check_val_every_n_epoch=max_epochs, logger=logger)
trainer.validate(model, dataloaders = loader_val )
trainer.fit(model,  train_dataloaders= loader_train, val_dataloaders= loader_val)
print("Test Result: ", trainer.test(dataloaders = loader_test) )


