# Building Energy Efficiency Prediction with Random Forest

## Overview
This project demonstrates how to generate a synthetic dataset of building features, train a Random Forest regressor to predict energy efficiency, visualize relationships, and interpret model feature importance.  
The environment setup uses **Ubuntu WSL** and **Jupyter Notebook**.

---

## Table of Contents
- [Prerequisites](#prerequisites)  
- [Environment Setup](#environment-setup)  
- [Running Jupyter Notebook](#running-jupyter-notebook)  
- [Installing Required Packages](#installing-required-packages)  
- [Project Code](#project-code)  
- [Common Issues and Solutions](#common-issues-and-solutions)  
- [Best Practices and Tips](#best-practices-and-tips)  

---

## Prerequisites

- Windows 10 or 11 with WSL installed  
- Basic familiarity with command line and Python recommended but not required  

---

## Environment Setup

### 1. Install WSL (Windows Subsystem for Linux)
If you haven’t already installed WSL, follow Microsoft’s official instructions:  
https://learn.microsoft.com/en-us/windows/wsl/install

### 2. Launch Ubuntu WSL Terminal

### 3. Update system packages

```bash
sudo apt update && sudo apt upgrade -y
```

### 4. Install Python, pip, and virtual environment tools

```bash
sudo apt install python3 python3-pip python3-venv -y
```

### 5. Create and activate a Python virtual environment
This isolates your Python setup and avoids conflicts.

```bash
python3 -m venv venv
source venv/bin/activate
```


### Installing Required Packages
Within the activated virtual environment, install all needed packages:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

### Running Jupyter Notebook
1. Start Jupyter Notebook server from WSL

```bash
jupyter notebook --allow-root --no-browser --ip=0.0.0.0
```


### 2. Copy the URL with the token shown in the terminal output, for example:

```bash
http://127.0.0.1:8888/?token=abc123def456...
```


### 3. Open that URL in your Windows web browser to access the Jupyter interface.

Project Code
Copy and paste the entire code below into a new notebook cell and run it:

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# Generate synthetic dataset for building features and energy efficiency ratings
np.random.seed(0)
data_size = 500
data = {
    'WallArea': np.random.randint(200, 400, data_size),
    'RoofArea': np.random.randint(100, 200, data_size),
    'OverallHeight': np.random.uniform(3, 10, data_size),
    'GlazingArea': np.random.uniform(0, 1, data_size),
    'EnergyEfficiency': np.random.uniform(10, 50, data_size)  # Energy efficiency rating
}
df = pd.DataFrame(data)

# Data preprocessing
X = df.drop('EnergyEfficiency', axis=1)
y = df['EnergyEfficiency']

# Visualize relationships between features and Energy Efficiency
sns.pairplot(df, x_vars=['WallArea', 'RoofArea', 'OverallHeight', 'GlazingArea'], y_vars='EnergyEfficiency', height=4, aspect=1, kind='scatter')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest regressor with fixed random state for reproducibility
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Plot True vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()

# Plot feature importances
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(8, 5))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
```


### Common Issues and Solutions
Issue 1: ModuleNotFoundError for packages like pandas or seaborn
You likely installed packages outside the virtual environment used by Jupyter.

Solution: Activate the same virtual environment used to launch Jupyter, then run:

```bash
pip install pandas seaborn
```

Restart your Jupyter kernel after installation.

### Issue 2: Jupyter refuses to run due to root user warning
Cause: Running WSL as root.

Solution: Add --allow-root flag:

```bash
jupyter notebook --allow-root --no-browser --ip=0.0.0.0
```

### Issue 3: Cannot access Jupyter URL from Windows browser
Make sure you copy the full URL with the token from WSL terminal.

```
Use http://127.0.0.1:8888/?token=... URL for best compatibility.
```


### Issue 4: Need to run installations but terminal is occupied by Jupyter
Keep two terminals open:

One running Jupyter server

One for activating environment and installing packages

Activate the same virtual environment in both terminals.

Best Practices and Tips
Always create and activate a virtual environment to isolate dependencies.

Use fixed random_state in models for reproducibility.

Restart notebook kernel after installing new packages.

Use clear and descriptive variable names and comments to make the code maintainable.

Consider using JupyterLab (jupyter lab) for a richer interface.

For large datasets, explore model tuning and cross-validation for better accuracy.

### Summary
This guide walks you through setting up a Python data science environment on Ubuntu WSL, running Jupyter notebooks, installing packages, writing and running a Random Forest regression model, visualizing data, and troubleshooting common issues.

If you follow these steps carefully, even without prior Python or Jupyter experience, you will be able to replicate and extend this project seamlessly.
