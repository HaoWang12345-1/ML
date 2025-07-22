# Multi-Objective Materials Optimization via Pareto Set Learning and Active Learning

This repository implements a complete machine learning framework for multi-objective optimization of polymer composites. It integrates:

- Gaussian Process (GP) Regression Models for property prediction
- Pareto Set Learning (PSL) using an MLP to map preference vectors to optimal formulations
- Active Learning (AL) to iteratively select informative experiments using LCB + Hypervolume Improvement strategies

## Folder Structure

.
├── gp_model.py               # GP regression model training and saving
├── pareto_set_learning.py   # PSL model (lambda → x mapping) and optimization
├── active_learning.py       # Active learning loop (LCB + HVI)
├── input_data.csv           # Experimental data (composition + performance)
├── requirements.txt         # Python dependency list
└── README.md                # Instructions and usage guide

## Installation

Create a new environment and install dependencies:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda create -n psl_env python=3.10
conda activate psl_env
pip install -r requirements.txt
```

## Execution Guide

Ensure the input file input_data.csv is placed in the same directory.

### Step 1: Train Gaussian Process Models

```bash
python gp_model.py
```

This script:
- Loads the experimental data from input_data.csv
- Trains three GP regressors for strength, fracture toughness, and impact toughness
- Saves trained models (gp_strength.pkl, gp_fracture.pkl, gp_impact.pkl) and scalers

### Step 2: Train Pareto Set Model via PSL

```bash
python pareto_set_learning.py
```

This script:
- Samples lambda vectors (preference vectors) from a Dirichlet distribution
- Optimizes the MLP model to learn lambda → x mapping using Augmented Tchebycheff loss
- Saves the trained model and outputs a .csv file with predicted optimal formulations and performances

### Step 3: Active Learning Optimization

```bash
python active_learning.py
```

This script:
- Loads the current GP and PSL models
- Selects high-value non-dominated points based on LCB + Hypervolume Improvement
- Saves newly selected points and updates the design set for the next round

## Output Files

- gp_strength.pkl, gp_fracture.pkl, gp_impact.pkl: Trained GP models
- psl_outputs.csv: PSL-generated optimal design set (composition + predicted performance)
- active_learning_round_i.csv: Selected points in each AL iteration


## License

This code is released for academic use under the MIT License.
