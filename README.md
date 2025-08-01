# Surrogate-Losses-for-Decision-Focused-Learning-using-Differentiable-Optimization
Code for the ECAi 2025 paper "Minimizing Surrogate Losses for Decision-Focused Learning using Differentiable Optimization"

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

## Modifying Configuration Parameters

To run experiments with different configurations, you can modify the parameter values directly in the Exp_Run.sh file:

1. For Shortest Path experiments, modify the `--grid_size` parameter (e.g., change from 15 to another value)
2. For Knapsack experiments, modify the `--num_items` parameter (e.g., change from 400 to another value)
3. For Facility Location experiments, modify the `--num_customers` parameter (e.g., change from 200 to another value) and `--num_facilities` if needed

Example of modifying grid size in the Exp_Run.sh file:
```bash
# Change this line
python3 ShortestPathExpBaseline.py --model_name SPO --grid_size 15 --seed $s

# To this for a different grid size
python3 ShortestPathExpBaseline.py --model_name SPO --grid_size 20 --seed $s
```

You can also add additional parameter configurations by duplicating and modifying the existing command lines.

## Hyperparameter Configurations

Important: The experiment scripts read hyperparameter configurations from the `pkg/configs` folder. These configuration files include:

- `shortestpath_config.json` and `shortestpath_DYSconfig.json`: For Shortest Path experiments
- `knapsack_config.json` and `knapsack_DYSconfig.json`: For Knapsack experiments
- `facilitylocation_config.json` and `facilitylocation_DYSconfig.json`: For Facility Location experiments
- `ESP_config.json` and `ESP_DYSconfig.json`: For ESP experiments

If you need to modify hyperparameters such as learning rates, model architectures, or optimization settings, edit these JSON configuration files rather than changing the command-line arguments in the Exp_Run.sh script.

Each experiment script will generate results that will be saved in the appropriate directories for later analysis and visualization.

Note: The experiments may take a significant amount of time to complete depending on the parameters and dataset sizes.
