import gurobipy as gp
from gurobipy import GRB
import numpy as np

# PyEPO library for end-to-end predict-then-optimize
import pyepo

# PyTorch and related imports
import torch
from torch import nn
from torch.utils.data import DataLoader

# Utilities and configuration
import argparse
import json
from pytorch_lightning.loggers import CSVLogger
import os
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def load_config(config_file):
    """Load configuration from file
    
    Args:
        config_file: Path to the JSON configuration file
        
    Returns:
        Dictionary containing the configuration parameters
    """
    with open(config_file, 'r') as f:
        return json.load(f)

def get_model_config(config, model_name, grid_size):
    """Get model-specific configuration"""
    common_params = config['common_params']
    model_config = config['model_configs'][model_name]
    
    # Check if the provided grid size is a default one
    if str(grid_size) in map(str, model_config['hyperparameters'].keys()):
        # Combine common params with model-specific params for this grid size
        full_config = {
            **common_params,
            **model_config['hyperparameters'][str(grid_size)],
            'model_name': model_name,
            'grid_size': grid_size
        }
    else:
        # Use default hyperparameters for grid sizes not explicitly defined
        full_config = {
            **common_params,
            'model_name': model_name,
            'grid_size': grid_size,
            'lr': 0.005,       # Default value, will be overridden if specified
            'sigma': 0.1,    # Default value, will be overridden if specified
            'n_samples': 1  # Default value, will be overridden if specified
        }
    return full_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="path to config file", default="configs/shortestpath_config.json")
parser.add_argument("--model_name", type=str, required=True, help="name of models (e.g., SPO, SCE, PFL)")
parser.add_argument("--grid_size", type=int, help="Size of the Grid", default=5)

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
parser.add_argument("--solve_ratio", type=float, help="solve ratio for caching")
parser.add_argument("--tau", type=float, help="Tau, the parameter of quadratic regularizer")
args = parser.parse_args()

# Load configuration from file and get model-specific settings
config = load_config(args.config)
config = get_model_config(config, args.model_name, args.grid_size)

# Override configuration with any command line arguments that were explicitly provided
# This allows for quick experimentation without editing config files
for key, value in vars(args).items():
    if key not in ['config', 'model_name', 'grid_size'] and value is not None:
        config[key] = value

# Set random seed for reproducibility
torch.manual_seed(config['seed'])

# Extract configuration parameters
grid_size, num_data, num_feat = config['grid_size'], config['num_data'], config['num_feat']
deg, noise_width = config['deg'], config['noise_width']
grid = (grid_size, grid_size)  # grid size

modelname = config['model_name']  # Model to use for training
normalize = config['normalize']  # Whether to normalize data

# Training parameters
batch_size, max_epochs = config['batch_size'], config['max_epochs']
lr, scheduler, seed = config['lr'], config['scheduler'], config['seed']

# Generate synthetic data for shortest path problem
# Features (feats) are used to predict edge costs (costs)
feats, costs = pyepo.data.shortestpath.genData(num_data+1000, num_feat, grid, deg, noise_width, seed=seed)

# Split data into train/validation/test sets
from sklearn.model_selection import train_test_split
# First split out test set (1000 samples)
x_train, x_test, c_train, c_test = train_test_split(feats, costs, test_size=1000, random_state=seed)
# Then split remaining data into train/validation
x_train, x_val, c_train, c_val = train_test_split(x_train, c_train, test_size=0.2, random_state=seed) 

# Create the optimization model for shortest path problem
optmodel = pyepo.model.grb.shortestPathModel((grid_size, grid_size))
# if modelname=='CAVE':
#     optmodel_train = pyepo.model.grb.shortestPathModelBinary( (grid_size,grid_size) )

# get optDataset
dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
dataset_val = pyepo.data.dataset.optDataset(optmodel, x_val, c_val)
dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers= 19)
loader_val  = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers= 19 )
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers= 19 )


import pytorch_lightning as pl
from LightningDFL_Models import  SPO, PFL, CAVE,  PFY,  SCE, CVX
from MLmodels import LinearRegression

log_dir = os.getcwd() + "/ResultECAI/ShortestPathResults/"

num_arcs = (grid[0]-1)*grid[1]+(grid[1]-1)*grid[0]
reg = LinearRegression(num_feat, num_arcs  ) # init model

if modelname=='SPO':
    solve_ratio = config['solve_ratio']
    
    logger = CSVLogger(log_dir, name='relaxspo_normalize{}_deg{}_noise{}_gridsize{}'.format(normalize, 
    deg, noise_width, grid_size))
    
    model = SPO(net= reg,lr=lr, scheduler=scheduler, max_epochs=max_epochs,
                seed=seed, optmodel= optmodel, normalize=normalize, 
                solve_ratio=solve_ratio, dataset=dataset_train)

if modelname=='SCE':
    solve_ratio = config['solve_ratio']
    logger = CSVLogger(log_dir, name='relaxSCE_normalize{}_deg{}_noise{}_gridsize{}'.format(normalize, 
            deg, noise_width, grid_size))

    model = SCE(net= reg,lr=lr, scheduler=scheduler, seed=seed, optmodel= optmodel, max_epochs=max_epochs,
            normalize=normalize, solve_ratio=solve_ratio, dataset=dataset_train)

if modelname=='PFL':
    logger = CSVLogger(log_dir, name='PFL_deg{}_noise{}_gridsize{}'.format(deg, noise_width, grid_size))
    model = PFL(net= reg, lr=lr, scheduler=scheduler, seed=seed, optmodel= optmodel, max_epochs=max_epochs)

if modelname=='PFY':
    n_samples  = config['n_samples']
    sigma  = config['sigma']
    logger = CSVLogger(log_dir, name='relaxpfy_deg{}_noise{}_gridsize{}'.format(deg, noise_width, grid_size))
    model = PFY(net= reg, n_samples=n_samples, sigma=sigma,  lr=lr,  max_epochs=max_epochs,
                scheduler=scheduler, seed=seed, optmodel= optmodel)

if modelname=='CAVE':
    from pyepo.data.dataset_util import collate_fn
    dataset_train = pyepo.data.dataset_util.optDatasetConstrs(optmodel, x_train, c_train)
    loader_train = DataLoader(dataset_train, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=19)

    logger = CSVLogger(log_dir, name='relaxcave_deg{}_noise{}_gridsize{}'.format(deg, noise_width, grid_size))
    model = CAVE(net= reg,  lr=lr, max_epochs=max_epochs, scheduler=scheduler, seed=seed, optmodel= optmodel)

if "CVX" in modelname:
    dflloss = config['dflloss']
    tau =  config['tau']
    logger = CSVLogger(log_dir, name='cvx{}_normalize{}_deg{}_noise{}_gridsize{}'.format(dflloss, normalize, deg, noise_width, grid_size))
    model = CVX(net= reg, dflloss = dflloss, tau = tau, max_epochs=max_epochs,
                 lr=lr, scheduler=scheduler, seed=seed, optmodel= optmodel)

print (f"Training of {modelname} to be started")
trainer = pl.Trainer(max_epochs=max_epochs,  logger=logger )
# trainer = pl.Trainer(max_epochs=max_epochs, check_val_every_n_epoch=max_epochs, logger=logger)
trainer.validate(model, dataloaders = loader_val )
trainer.fit(model,  train_dataloaders= loader_train, val_dataloaders= loader_val)
print (trainer.predict(dataloaders=loader_val))
print("Test Result: ", trainer.test(dataloaders = loader_test) )
