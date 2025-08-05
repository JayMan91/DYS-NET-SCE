# Surrogate-Losses-for-Decision-Focused-Learning-using-Differentiable-Optimization

Code for the ECAI 2025 paper "Minimizing Surrogate Losses for Decision-Focused Learning using Differentiable Optimization"

## Summary

This repository contains the implementation of our novel approach to Decision-Focused Learning (DFL) using differentiable optimization layer [DYS-Net](https://github.com/mines-opt-ml/fpo-dys). We explore various surrogate losses and demonstrate their effectiveness across multiple LP/ILP/MILP optimization problems.

## Acknowledgments

*Our implementation builds upon [PyEPO](https://github.com/khalil-research/PyEPO), a benchmarking library for End-to-End Predict-then-Optimize techniques.*

## Installation

We recommend using a virtual environment to avoid conflicts with other Python packages:

```bash
# Create a virtual environment
python3 -m venv env_dfl

# Activate the virtual environment
# On Linux/Mac
source env_dfl/bin/activate
# On Windows
# env_dfl\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Instructions for Running Experiments

To run the experiments, execute the Exp_run.sh script from the command line:

```bash
./Exp_run.sh
```

Make sure the script has executable permissions:

```bash
chmod +x Exp_run.sh
```

The Exp_run.sh script will execute all the necessary experiment scripts in the correct order, running various models with different parameters.
### Model Name in Exp_run.sh

Here's how the math notation in our paper translates to actual model names in the code:

- CVX Models
   - $Regret^{CVX}$ corresponds to `CVX-Regret`
   - $SqDE^{CVX}$ corresponds to `CVX-Squared`
   - $SPO_{+}^{CVX}$ corresponds to `CVX-SPO`
   - $SCE^{CVX}$ corresponds to `CVX-SCE`

- DYS-Net Models
   - $Regret^{DYS}$ corresponds to `DYS-Regret`
   - $SqDE^{DYS}$ corresponds to `DYS-Squared`
   - $SPO_{+}^{DYS}$ corresponds to `DYS-SPO`
   - $SCE^{DYS}$ corresponds to `DYS-SCE`
## Modifying Configuration Parameters

To run experiments with different configurations, you can modify the parameter values directly in the Exp_Run.sh file:

1. For Shortest Path experiments, modify the `--grid_size` parameter (e.g., change from 15 to another value)
2. For Knapsack experiments, modify the `--num_items` parameter (e.g., change from 400 to another value)
3. For Facility Location experiments, modify the `--num_customers` parameter (e.g., change from 200 to another value) and `--num_facilities` if needed

## Hyperparameter Configurations

Important: The experiment scripts read hyperparameter configurations from the `pkg/configs` folder. These configuration files include:

- `shortestpath_config.json` and `shortestpath_DYSconfig.json`: For Shortest Path experiments
- `knapsack_config.json` and `knapsack_DYSconfig.json`: For Knapsack experiments
- `facilitylocation_config.json` and `facilitylocation_DYSconfig.json`: For Facility Location experiments

If you need to modify hyperparameters such as learning rates, model architectures, or optimization settings, edit these JSON configuration files rather than changing the command-line arguments in the Exp_Run.sh script.



## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{surrogate-losses-dfl-2025,
  title={Minimizing Surrogate Losses for Decision-Focused Learning using Differentiable Optimization},
  author={[Author Names]},
  booktitle={Proceedings of the European Conference on Artificial Intelligence (ECAI)},
  year={2025}
}
