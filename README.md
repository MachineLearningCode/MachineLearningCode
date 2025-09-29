# Gas Concentration Prediction Using Machine Learning

This repository contains code for training and evaluating multiple machine learning models to predict gas emissions based on 9 input features. The models include Multilayer Perception (MLP), Support Vector Regression (SVR), k-Nearest Neighbor (kNN), and Random Forest (RF). The project also includes performance visualization and feature importance analysis using SHAP.

## 📁 Folder Structure

- `models/`: Trained models for each gas and model type.
- `plots/`: 
  - `bar_comparison.png`: Bar plots comparing predicted vs. actual values.
  - `prediction_plot.png`: Scatter plots showing predicted vs. actual values.
  - `shap_importance.png`: SHAP summary plots for feature importance.
  - `shap_importance_bar.png`: SHAP bar plots.
- `results/`: CSV files of test predictions for each model and gas.

## 🧠 Models

Four regressors are implemented:
- `MLP`
- `SVR`
- `kNN`
- `RF`

Each is trained separately on the gases: `NH₃`, `CH₄`, `CO₂`, and `N₂O`.

## 🧪 Dataset

The dataset is in a file called `data.csv` with the following columns:

### Input Features:
- `C/N`, `OM`, `Temperature`, `MC`, `PH`, `TN`, `NH₄⁺-N`, `NO₃⁻-N mg/kg`, `EC`

### Target Gases:
- `NH₃g/m²·d`, `CH₄g/m²·d`, `CO₂g/m²·d`, `N₂Og/m²·d`

### Special Column:
- `Mo`: Used to split the data into training (`Mo=0`) and test (`Mo=1`) sets.

## 🧪 Additional Application Dataset

In addition to the main dataset (data.csv), which contains 350 samples used for model training and evaluation, we prepared a separate file named experiment_data.csv. This file includes only our real-world experimental measurements collected under nanomembrane-covered composting conditions. Unlike data.csv, the experiment_data.csv file was not used during training or testing of the models. Instead, it was reserved exclusively for model application, where the trained MLP was applied to assess the mitigation effect of the nanomembrane by projecting uncovered emissions from covered-condition input features.

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

Follow these steps to execute the project:

### 1. Prepare the Dataset

Place your dataset in the root directory

### 2. Install Dependencies

Ensure you have Python3 installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Exceut the main script:
python all_models.py

### 4. Result
After running the script, you’ll find:

Trained models in: models/

Scaler saved in: models/feature_scaler.joblib

Evaluation plots in: plots/

Test predictions in: results/

Each model and gas is handled separately for modular analysis and inspection.

