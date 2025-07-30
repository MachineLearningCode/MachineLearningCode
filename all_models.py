import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os
import shap

# Create directories for saving models and plots
os.makedirs('all_models_mo_split/models', exist_ok=True)
os.makedirs('all_models_mo_split/plots', exist_ok=True)
os.makedirs('all_models_mo_split/results', exist_ok=True)

# Load the data
data = pd.read_csv('data.csv')
font = 20
# Define input features and target variables
input_features = ['C/N', 'SOM', 'Temperature', 'water contentration', 'PH', 'TN', 'NH4', 'NO3', 'EC']
target_variables = ['$NH_3$', '$CH_4$', '$CO_2$', '$N_2O$']

# Split data based on Mo value
X_train = data[data['Mo'] == 0][input_features]
X_test = data[data['Mo'] == 1][input_features]
y_train = data[data['Mo'] == 0][target_variables]
y_test = data[data['Mo'] == 1][target_variables]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#save the scaler
joblib.dump(scaler, 'all_models_mo_split/models/feature_scaler.joblib')

X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
    X_train_scaled, y_train,
    test_size=0.2,
    random_state=42
)

# Define models and their parameters
models_config = {
    'MLP': {
        'model': MLPRegressor,
        'params': {
            'hidden_layer_sizes': (128, 256, 128),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 1000000,
            'random_state': 42
        }
    },
    'SVR': {
        'model': SVR,
        'params': {
            'kernel': 'rbf',
            'C': 100,
            'epsilon': 0.1
        }
    },
    'kNN': {
        'model': KNeighborsRegressor,
        'params': {
            'n_neighbors': 21,
            'weights': 'distance',
            'p': 2
        }
    },
    'RF': {
        'model': RandomForestRegressor,
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'max_features': 'log2',
            'random_state': 11
        }
    }
}

# Create dictionaries to store results
all_models = {}
all_predictions = {}
all_r2_scores = {}
all_rmse_scores = {}
all_mae_scores = {}
# Train and evaluate each model type
for model_name, config in models_config.items():
    print(f"\nTraining {model_name} models...")
    
    # Create directory for this model type
    model_dir = f'all_models_mo_split/models/{model_name.lower().replace(" ", "_")}'
    plot_dir = f'all_models_mo_split/plots/{model_name.lower().replace(" ", "_")}'
    results_dir = f'all_models_mo_split/results/{model_name.lower().replace(" ", "_")}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize dictionaries for this model type
    all_models[model_name] = {}
    all_predictions[model_name] = {}
    all_r2_scores[model_name] = {}
    all_rmse_scores[model_name] = {}
    all_mae_scores[model_name] = {}
    for target in target_variables:
        print(f"\nTraining {model_name} model for {target}")
        # Create and train model
        model = config['model'](**config['params'])
        if target == '$N_2O$':
          y_train[target] = y_train[target]*100
        model.fit(X_train_scaled, y_train[target])
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_pred = np.abs(y_pred)
        
        if target == '$N_2O$':
          y_pred = y_pred/100
          y_train_pred = y_train_pred/100
          y_val_pred = y_val_pred/100
          y_train[target] = y_train[target]/100
    
        # Calculate metrics using validation data
        r2 = r2_score(y_val[target], y_val_pred)
        mse = mean_squared_error(y_val[target], y_val_pred)
        mae = mean_absolute_error(y_val[target], y_val_pred)
        rmse = np.sqrt(mse)
        
        # Merge train and validation predictions and actual values
        y_train_val_pred = np.concatenate([y_train_pred, y_val_pred])
        y_train_val = np.concatenate([y_train[target], y_val[target]])
        
        # Calculate combined metrics
        r2t = r2_score(y_train[target], y_train_pred)
        mset = mean_squared_error(y_train[target], y_train_pred)
        maet = mean_absolute_error(y_train[target], y_train_pred)
        rmset = np.sqrt(mset)

        all_models[model_name][target] = model
        all_predictions[model_name][target] = y_pred
        all_r2_scores[model_name][target] = r2
        all_rmse_scores[model_name][target] = rmse
        all_mae_scores[model_name][target] = mae
        
        # Create bar plot comparing predicted vs actual values
        plt.figure(figsize=(12, 6))
        x = np.arange(len(y_test[target]))
        width = 0.35
        
        plt.bar(x - width/2, y_test[target], width, label='Actual', color='blue', alpha=0.7)
        plt.bar(x + width/2, y_pred, width, label='Predicted', color='red', alpha=0.7)
        
        plt.xlabel('Sample Index', fontsize=font)
        plt.ylabel(f'{target} Value', fontsize=font)
        plt.title(f'Actual vs Predicted {target} ({model_name})\nMo=1 Test Data', fontsize=20)
        plt.legend(fontsize=font)
        plt.xticks(x, fontsize=font)
        plt.yticks(fontsize=font)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{target}_bar_comparison.png')
        plt.close()
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'Sample_Index': range(len(y_test[target])),
            **{f'Input_{feature}': X_test[feature].values for feature in input_features},
            'Actual_Value': y_test[target].values,
            'Predicted_Value': y_pred
        })
        results_df.to_csv(f'{results_dir}/{target}_results.csv', index=False)
        # Plot actual vs predicted
        plt.figure(figsize=(9, 6))
        plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
        plt.scatter(y_train_val, y_train_val_pred, alpha=0.5, label='Train Data')
        plt.scatter(y_val[target], y_val_pred, alpha=1, label='Validation Data')
        plt.plot([min(y_test[target].min(), y_train[target].min()), 
                 max(y_test[target].max(), y_train[target].max())], 
                [min(y_test[target].min(), y_train[target].min()), 
                 max(y_test[target].max(), y_train[target].max())], 
                'r--', lw=2)
        plt.xlabel(f'Actual {target} Values ($g/(m^2*d)$)', fontsize=font)
        plt.ylabel(f'Predicted {target} Values ($g/(m^2*d)$)', fontsize=font)
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        plt.title(f'Actual vs Predicted {target} ({model_name})', fontsize=20)
        y_target = np.concatenate([y_train[target], y_val[target]])
        plt.text(6*max(y_target)/8, max(y_train_val_pred)*.2, f'$R^2$ train: {r2t:.4f}', fontsize=font, ha='center', va='center')
        plt.text(6*max(y_target)/8, max(y_train_val_pred)*.1, f'$R^2$ val: {r2:.4f}', fontsize=font, ha='center', va='center')
        plt.legend(fontsize=font)
        plt.savefig(f'{plot_dir}/{target}_prediction_plot.png')
        plt.close()
        
        # Save model
        joblib.dump(model, f'{model_dir}/{target}_model.joblib')
        
        # Plot SHAP values for feature importance
        if hasattr(model, 'predict'):
            # Create SHAP explainer based on model type
            if isinstance(model, (RandomForestRegressor)):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, X_train_scaled[:100])
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_val_scaled)
            
            # Create and save SHAP summary plot
            plt.figure()
            shap.summary_plot(shap_values, X_val_scaled, feature_names=input_features, 
                            show=False, plot_size=(10,8))
            plt.title(f'SHAP Feature Importance for {target} ({model_name})')
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{target}_shap_importance.png')
            plt.close()
            
            # Create and save SHAP bar plot
            plt.figure()
            shap.summary_plot(shap_values, X_val_scaled, feature_names=input_features,
                            plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance (Bar) for {target} ({model_name})')
            plt.tight_layout() 
            plt.savefig(f'{plot_dir}/{target}_shap_importance_bar.png')
            plt.close()
            
            # Print mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            print(f"\nSHAP Feature Importance for {target} ({model_name}):")
            for feature, importance in zip(input_features, mean_shap):
                print(f"{feature}: {importance:.4f}")
            # Plot feature importance using model's feature importances
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [input_features[i] for i in sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance for {target} ({model_name})\nMo=0 Training, Mo=1 Testing')
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{target}_feature_importance.png')
            plt.close()
            
            # Print feature importance values
            print(f"\nFeature Importance for {target} ({model_name}):")
            for feature, importance in zip(input_features, feature_importance):
                print(f"{feature}: {importance:.4f}")

# Save the scaler for future use
joblib.dump(scaler, 'all_models_mo_split/models/feature_scaler.joblib')

# Print comprehensive summary
print("\nComprehensive Model Summary:")
for model_name in models_config.keys():
    print(f"\n{model_name} Models:")
    for target in target_variables:
        print(f"\n{target}:")
        print(f"R2 Score: {all_r2_scores[model_name][target]:.4f}")
        print(f"RMSE: {all_rmse_scores[model_name][target]:.4f}")
        print(f"MAE: {all_mae_scores[model_name][target]:.4f}")

# Create comparison plots for each target variable
for target in target_variables:
    plt.figure(figsize=(12, 8))
    for model_name in models_config.keys():
        y_pred = all_predictions[model_name][target]
        plt.scatter(y_test[target], y_pred, alpha=0.5, label=f'{model_name} (R2: {all_r2_scores[model_name][target]:.4f})')
    
    plt.plot([y_test[target].min(), y_test[target].max()], 
             [y_test[target].min(), y_test[target].max()], 
             'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Model Comparison for {target}\nMo=0 Training, Mo=1 Testing')
    plt.legend()
    plt.savefig(f'all_models_mo_split/plots/{target}_model_comparison.png')
    plt.close() 