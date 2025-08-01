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
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def load_config(config_file):
    """Load configuration from file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def get_model_config(config, model_name, num_customers, num_facilities):
    """Get model-specific configuration"""
    common_params = config['common_params']
    model_config = config['model_configs'][model_name]
    
    # Check if both customer and facility sizes are in hyperparameters
    key = f"{num_customers}_{num_facilities}"
    if key in map(str, model_config['hyperparameters'].keys()):
        # Combine common params with model-specific params for this size
        full_config = {
            **common_params,
            **model_config['hyperparameters'][key],
            'model_name': model_name,
            'num_customers': num_customers,
            'num_facilities': num_facilities
        }
    else:
        # Use default hyperparameters
        full_config = {
            **common_params,
            'model_name': model_name,
            'num_customers': num_customers,
            'num_facilities': num_facilities,
            'lr': 0.005,       # Default value, will be overridden if specified
            'sigma': 0.1,    # Default value, will be overridden if specified
            'n_samples': 1  # Default value, will be overridden if specified
        }
    return full_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="path to config file", default="configs/facilitylocation_DYSconfig.json")
parser.add_argument("--model_name", type=str, required=True, help="name of models")
parser.add_argument("--num_customers", type=int, help="Number of Customers", default=20)
parser.add_argument("--num_facilities", type=int, help="Number of Facilities", default=5)
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
parser.add_argument("--alpha", type=float, help="the parameter of DYS")
parser.add_argument("--decay_param", type=float, help="the parameter of DYS DECAY")
parser.add_argument("--numiter", type=int, help="Number of DYS Iterations")

args = parser.parse_args()

# Load and get model-specific configuration
config = load_config(args.config)
config = get_model_config(config, args.model_name, args.num_customers, args.num_facilities)

# Update config with command line arguments (only if they are provided)
for key, value in vars(args).items():
    if key not in ['config', 'model_name', 'num_customers', 'num_facilities'] and value is not None:
        config[key] = value

# Set random seed
torch.manual_seed(config['seed'])

num_customers, num_facilities = config['num_customers'], config['num_facilities']
num_data, num_feat = config['num_data'], config['num_feat']
deg, noise_width = config['deg'], config['noise_width']

modelname = config['model_name']
normalize = config['normalize']


batch_size, max_epochs = config['batch_size'], config['max_epochs']
lr, scheduler, seed = config['lr'], config['scheduler'], config['seed']
test_size = 1000
demands, capacities, setup_costs, feats, costs = pyepo.data.facilitylocation.genData(
    num_data + test_size, num_feat, num_customers, num_facilities, deg, noise_width, seed=seed)

from sklearn.model_selection import train_test_split

x_train, x_test, c_train, c_test = train_test_split(feats, costs, test_size=test_size, random_state=seed)
x_train, x_val, c_train, c_val  = train_test_split(x_train, c_train, test_size=0.2, random_state=seed) 

optmodel = pyepo.model.grb.FacilityLocationModel(demands, capacities, setup_costs)

optmodel_train = optmodel

# get optDataset
dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
dataset_val = pyepo.data.dataset.optDataset(optmodel, x_val, c_val)
dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=19)
loader_val  = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=19)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=19)
import pytorch_lightning as pl
from LightningDFL_Models import  DYS
from MLmodels import LinearRegression

log_dir = os.getcwd() + "/ResultECAI/FacilityLocationResults/"

num_arcs = num_facilities * num_customers
reg = LinearRegression(num_feat, num_arcs  ) # init model


from FacilityLocation_utils import cost_transform, sol_transform
c_transform = cost_transform(optmodel)
s_transform = sol_transform(optmodel)

if "DYS" in modelname:
    dflloss = config['dflloss']
    tau =  config['tau']
    alpha = config['alpha']
    decay =  config['decay_param']
    numiter = config['numiter']

    
    logger = CSVLogger(log_dir, name='dys{}_normalize{}_deg{}_noise{}_customers{}_facilities{}'.format(
        dflloss, normalize, deg, noise_width, num_customers, num_facilities))
    model = DYS(net=reg, dflloss=dflloss,
                tau=tau,   alpha = alpha,decay_param = decay, num_iter=numiter,
                doScale = False, max_epochs=max_epochs,
                cost_transform= c_transform, sol_transform= s_transform,
                lr=lr, scheduler=scheduler, seed=seed, optmodel=optmodel)
print("\n" + "="*50)
print(f"Training of {modelname} to be started")
print("="*50 + "\n")

trainer = pl.Trainer(max_epochs=max_epochs, check_val_every_n_epoch=max_epochs, logger=logger)
trainer.validate(model, dataloaders=loader_val)
trainer.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_val)
trainer.test(dataloaders=loader_test)
