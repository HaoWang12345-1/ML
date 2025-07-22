import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Code Ocean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import joblib
import os
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for data file and output directory."""
    parser = argparse.ArgumentParser(description='Pareto Set Model with Gaussian Process Regression')
    parser.add_argument('--data_file', type=str, default='input_data.csv',
                        help='Path to input CSV file with data (optional, uses hard-coded data if absent)')
    parser.add_argument('--output_dir', type=str, default='/results',
                        help='Directory to save output files')
    return parser.parse_args()

# Disable LaTeX rendering
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

# Step 1: Data Preparation
def load_data(data_file, remove_outliers=True):
    """
    Load data from CSV or use hard-coded data, remove outliers (samples 2, 6, 20, 22, 24, 25, 26, 28, 29) by default.
    Load pre-trained scalers and polynomial transformer.
    """
    logger.info("Loading data...")
    if os.path.exists(data_file):
        logger.info(f"Loading data from {data_file} with encoding utf-8...")
        try:
            df = pd.read_csv(data_file, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"Failed to read {data_file} with utf-8 encoding. Trying 'utf-8-sig'...")
            try:
                df = pd.read_csv(data_file, encoding='utf-8-sig')
            except UnicodeDecodeError:
                logger.warning(f"Failed to read {data_file} with utf-8-sig encoding. Trying 'cp1252'...")
                try:
                    df = pd.read_csv(data_file, encoding='cp1252')
                except UnicodeDecodeError as e:
                    logger.error(f"Cannot decode {data_file}. Tried UTF-8, UTF-8-SIG, and CP1252. Falling back to hard-coded data.")
                    df = None
        except Exception as e:
            logger.error(f"Failed to load {data_file}: {e}. Falling back to hard-coded data.")
            df = None
    else:
        logger.info(f"No data file found at {data_file}. Using hard-coded data.")
        df = None

    if df is None:
        data = {
            'TPU (%)': [0.79, 2.08, 1.17, 4.18, 1.7, 0.91, 4.04, 3.3, 1.43, 0.25, 4.98, 3.16, 0.46, 0.12, 2.58, 1.61, 3.41, 4.71, 2.49, 2.23, 3.73, 2.88, 1.14, 1.98, 4.58, 0.52, 4.37, 3.58, 3.89, 2.71],
            'BF (%)': [10.25, 9.89, 7.2, 1.14, 4.19, 3.32, 12.84, 8.2, 11.81, 6.6, 3.65, 14.83, 5.76, 10.97, 9.89, 5.08, 11.49, 2.52, 1.77, 2.07, 8.7, 4.97, 0.53, 6.51, 12.09, 7.9, 13.83, 14.19, 13.23, 0.49],
            'Strength (MPa)': [217, 212, 202, 110, 144, 136, 230, 204, 228, 167, 139, 209, 158, 224, 219, 151, 226, 133, 122, 130, 206, 150, 81, 169, 229, 203, 236, 210, 232, 74],
            'Fracture Toughness (MPa·m^1/2)': [12.6, 14.1, 11.0, 6.1, 6.8, 6.4, 12.6, 13.4, 12.8, 8.1, 6.6, 6.2, 7.3, 11.2, 14.8, 7.1, 12.5, 6.3, 5.8, 6.8, 13.9, 7.3, 6.0, 8.3, 12.0, 11.6, 6.8, 6.6, 8.2, 5.5],
            'Impact Energy Dissipation (J)': [3.4, 4.3, 3.6, 2.6, 3.0, 2.9, 3.8, 3.6, 3.9, 3.3, 2.7, 2.7, 3.1, 3.4, 4.8, 3.2, 3.7, 2.8, 2.7, 2.9, 4.3, 3.1, 2.4, 3.4, 3.4, 3.9, 3.1, 2.8, 3.2, 2.2]
        }
        df = pd.DataFrame(data)

    if remove_outliers:
        logger.info("Removing outliers (samples 2, 6, 20, 22, 24, 25, 26, 28, 29)")
        df = df.drop(index=[2, 6, 20, 22, 24, 25, 26, 28, 29])
    X = df[['TPU (%)', 'BF (%)']].values
    interaction = (df['TPU (%)'] * df['BF (%)']).values.reshape(-1, 1)
    
    # Load scaler_interaction
    scaler_interaction_pkl = '/results/scaler_interaction.pkl'
    if not os.path.exists(scaler_interaction_pkl):
        logger.error(f"Scaler file {scaler_interaction_pkl} not found.")
        raise FileNotFoundError(f"Scaler file {scaler_interaction_pkl} not found.")
    logger.info(f"Loading scaler_interaction from {scaler_interaction_pkl}...")
    scaler_interaction = joblib.load(scaler_interaction_pkl)
    X_interaction = scaler_interaction.transform(interaction)
    
    # Load poly and scaler_poly
    poly_pkl = '/results/poly.pkl'
    scaler_poly_pkl = '/results/scaler_poly.pkl'
    if not os.path.exists(poly_pkl):
        logger.error(f"Polynomial transformer file {poly_pkl} not found.")
        raise FileNotFoundError(f"Polynomial transformer file {poly_pkl} not found.")
    if not os.path.exists(scaler_poly_pkl):
        logger.error(f"Scaler file {scaler_poly_pkl} not found.")
        raise FileNotFoundError(f"Scaler file {scaler_poly_pkl} not found.")
    logger.info(f"Loading poly from {poly_pkl} and scaler_poly from {scaler_poly_pkl}...")
    poly = joblib.load(poly_pkl)
    scaler_poly = joblib.load(scaler_poly_pkl)
    
    X_poly = poly.transform(X)
    X_poly_scaled = scaler_poly.transform(X_poly)
    X = np.column_stack((X_poly_scaled, X_interaction))
    y = df[['Strength (MPa)', 'Fracture Toughness (MPa·m1/2)', 'Impact Energy Dissipation (J)']].values
    df['POK (%)'] = 100 - df['TPU (%)'] - df['BF (%)']
    
    # Verify POK constraint
    invalid_pok = df[df['POK (%)'] < 80]
    if not invalid_pok.empty:
        logger.warning(f"Found {len(invalid_pok)} samples with POK < 80: {invalid_pok['POK (%)'].values}")
    
    return X, y, df.index.values, scaler_interaction, scaler_poly, poly, df[['TPU (%)', 'BF (%)', 'POK (%)']].values

def preprocess_data(X, y, output_dir):
    """
    Scale inputs and log-transform outputs using pre-trained scalers.
    """
    logger.info("Preprocessing data...")
    
    # Load scaler_X
    scaler_X_pkl = os.path.join(output_dir, "scaler_X.pkl")
    if not os.path.exists(scaler_X_pkl):
        logger.error(f"Scaler file {scaler_X_pkl} not found.")
        raise FileNotFoundError(f"Scaler file {scaler_X_pkl} not found.")
    logger.info(f"Loading scaler_X from {scaler_X_pkl}...")
    scaler_X = joblib.load(scaler_X_pkl)
    X_scaled = scaler_X.transform(X)
    
    # Log-transform outputs
    y_log = np.log(y + 1e-1) * np.sign(y + 1e-1)
    
    # Load individual scalers for each output
    scaler_strength_pkl = os.path.join(output_dir, "scaler_strength.pkl")
    scaler_fracture_pkl = os.path.join(output_dir, "scaler_fracture.pkl")
    scaler_impact_pkl = os.path.join(output_dir, "scaler_impact.pkl")
    for pkl_file in [scaler_strength_pkl, scaler_fracture_pkl, scaler_impact_pkl]:
        if not os.path.exists(pkl_file):
            logger.error(f"Scaler file {pkl_file} not found.")
            raise FileNotFoundError(f"Scaler file {pkl_file} not found.")
    
    logger.info(f"Loading scalers: {scaler_strength_pkl}, {scaler_fracture_pkl}, {scaler_impact_pkl}...")
    scaler_strength = joblib.load(scaler_strength_pkl)
    scaler_fracture = joblib.load(scaler_fracture_pkl)
    scaler_impact = joblib.load(scaler_impact_pkl)
    
    y_scaled_strength = scaler_strength.transform(y_log[:, 0].reshape(-1, 1)).ravel()
    y_scaled_fracture = scaler_fracture.transform(y_log[:, 1].reshape(-1, 1)).ravel()
    y_scaled_impact = scaler_impact.transform(y_log[:, 2].reshape(-1, 1)).ravel()
    
    return X_scaled, [y_scaled_strength, y_scaled_fracture, y_scaled_impact], scaler_X, [scaler_strength, scaler_fracture, scaler_impact]

def inverse_transform_y(y_pred_scaled, scaler, idx):
    """
    Inverse transform scaled and log-transformed output for a single objective.
    """
    y_log = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_unscaled = np.exp(y_log * np.sign(y_log)) - 1e-1
    return y_unscaled

# Step 2: Load Gaussian Process Models
def load_gp_models(output_dir):
    """
    Load pre-trained GP models from .pkl files.
    """
    logger.info("Loading pre-trained GP models...")
    gp_models = []
    objectives = ['strength', 'fracture', 'impact']
    for obj in objectives:
        pkl_file = os.path.join(output_dir, f"gp_{obj}.pkl")
        if not os.path.exists(pkl_file):
            logger.error(f"GP model file {pkl_file} not found.")
            raise FileNotFoundError(f"GP model file {pkl_file} not found.")
        try:
            gp_model = joblib.load(pkl_file)
            gp_models.append(gp_model)
            logger.info(f"Loaded GP model for {obj} from {pkl_file}")
        except Exception as e:
            logger.error(f"Failed to load GP model for {obj} from {pkl_file}: {e}")
            raise
    logger.info(f"Successfully loaded {len(gp_models)} GP models.")
    return gp_models

# Step 3: Predict with GP
def predict_gp(gp_models, X, scalers, original_y):
    """
    Predict with GP models.
    """
    logger.info("Predicting with GP models...")
    means, stds = [], []
    for i, (gp, scaler) in enumerate(zip(gp_models, scalers)):
        mean, std = gp.predict(X, return_std=True)
        means.append(inverse_transform_y(mean, scaler, i))
        stds.append(std * scaler.scale_)
    y_pred = np.column_stack(means)
    y_std = np.column_stack(stds)
    return y_pred, y_std

# Step 4: Cross-Validation for GP Validation
def cross_validate_gp(X_scaled, y_scaled, scalers, original_y, indices, output_dir, n_splits=5):
    """
    Perform k-fold cross-validation with outlier detection and R² validation.
    """
    logger.info("Performing cross-validation...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = {0: [], 1: [], 2: []}
    r2_scores = {0: [], 1: [], 2: []}
    coverage_scores = {0: [], 1: [], 2: []}
    residuals = {0: [], 1: [], 2: []}
    outlier_indices = {0: [], 1: [], 2: []}
    objectives = ['Strength (MPa)', 'Fracture Toughness (MPa·m^1/2)', 'Impact Energy Dissipation (J)']
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        logger.info(f"Processing fold {fold_idx+1}/{n_splits}")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = [y_scaled[i][train_idx] for i in range(3)]
        y_test = [y_scaled[i][test_idx] for i in range(3)]
        y_test_orig = original_y[test_idx]
        
        # Load fold-specific GP models
        gp_models = []
        for obj in ['strength', 'fracture', 'impact']:
            pkl_file = os.path.join(output_dir, f"gp_{obj}_fold_{fold_idx+1}.pkl")
            if not os.path.exists(pkl_file):
                logger.error(f"Fold-specific GP model file {pkl_file} not found.")
                raise FileNotFoundError(f"Fold-specific GP model file {pkl_file} not found.")
            try:
                gp_model = joblib.load(pkl_file)
                gp_models.append(gp_model)
                logger.info(f"Loaded GP model for {obj} fold {fold_idx+1} from {pkl_file}")
            except Exception as e:
                logger.error(f"Failed to load GP model for {obj} fold {fold_idx+1} from {pkl_file}: {e}")
                raise
        
        # Predict and validate
        y_pred, y_std = predict_gp(gp_models, X_test, scalers, original_y)
        mse = np.mean((y_pred - y_test_orig) ** 2, axis=0)
        r2 = [r2_score(y_test_orig[:, i], y_pred[:, i]) for i in range(3)]
        coverage = [np.mean((y_test_orig[:, i] >= y_pred[:, i] - 1.96 * y_std[:, i]) & 
                           (y_test_orig[:, i] <= y_pred[:, i] + 1.96 * y_std[:, i])) for i in range(3)]
        res = y_pred - y_test_orig
        
        for i in range(3):
            mse_scores[i].append(mse[i])
            r2_scores[i].append(r2[i])
            coverage_scores[i].append(coverage[i])
            residuals[i].append(res[:, i])
            threshold = [20, 2, 0.5][i]
            outliers = indices[test_idx][np.abs(res[:, i]) > threshold]
            outlier_indices[i].extend(outliers)
            if r2[i] < 0.82:
                logger.warning(f"{objectives[i]} R² ({r2[i]:.3f}) in fold {fold_idx+1} is below 0.82")
    
    avg_mse = {i: np.mean(mse_scores[i]) for i in range(3)}
    avg_r2 = {i: np.mean(r2_scores[i]) for i in range(3)}
    avg_coverage = {i: np.mean(coverage_scores[i]) for i in range(3)}
    max_residuals = {i: np.max(np.abs(np.concatenate(residuals[i]))) for i in range(3)}
    for i, obj in enumerate(objectives):
        print(f"Average MSE for {obj}: {avg_mse[i]:.4f}")
        print(f"Average R² for {obj}: {avg_r2[i]:.4f}")
        print(f"Average 95% CI Coverage for {obj}: {avg_coverage[i]:.4f}")
        print(f"Max Absolute Residual for {obj}: {max_residuals[i]:.4f}")
        print(f"Outliers for {obj} (cross-validation, |residual| > {[20, 2, 0.5][i]}): {list(set(outlier_indices[i]))}")
    
    # Save cross-validation results
    results = pd.DataFrame({
        'Metric': objectives,
        'Average MSE': [avg_mse[i] for i in range(3)],
        'Average R2 Score': [avg_r2[i] for i in range(3)],
        'Average 95% CI Coverage': [avg_coverage[i] for i in range(3)],
        'Max Absolute Residual': [max_residuals[i] for i in range(3)]
    })
    results.to_csv(os.path.join(output_dir, 'gp_results.csv'), index=False, encoding='utf-8')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, obj in enumerate(objectives):
        res = np.concatenate(residuals[i])
        actual = np.concatenate([original_y[test_idx][:, i] for _, test_idx in kf.split(X_scaled)])
        axes[i].scatter(actual, res, c='blue', alpha=0.5)
        axes[i].axhline(0, color='red', linestyle='--')
        axes[i].set_xlabel(f'Actual {obj}')
        axes[i].set_ylabel('Residual (Predicted - Actual)')
        axes[i].set_title(f'Residuals for {obj}')
        axes[i].grid(True)
    plt.savefig(os.path.join(output_dir, "residuals.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return avg_mse, avg_r2, avg_coverage, residuals

# Step 5: MLP Set Model
class ParetoSetModel(nn.Module):
    """
    MLP to map preference vectors to Pareto solutions.
    """
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=128):
        super(ParetoSetModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Step 6: Augmented Tchebycheff Scalarization with Target Penalty
def augmented_tch_scalarization(f_pred, lambda_vec, z_star, rho=0.01, epsilon_factor=0.05, target_region=None):
    """
    Compute augmented TCH scalarization with strong penalty.
    """
    epsilon = epsilon_factor * np.abs(z_star)
    utopia = z_star - epsilon
    weighted_diff = lambda_vec * (f_pred - utopia)
    max_term = np.max(weighted_diff)
    aug_term = rho * np.sum(lambda_vec * f_pred)
    
    if target_region is not None:
        target_min = target_region['min']
        target_max = target_region['max']
        penalty = 0
        for i in range(len(f_pred)):
            if not (target_min[i] <= f_pred[i] <= target_max[i]):
                penalty += 500 * max(abs(f_pred[i] - target_min[i]), abs(f_pred[i] - target_max[i]))
        return max_term + aug_term + penalty
    return max_term + aug_term

# Step 7: Train Pareto Set Model
def train_pareto_set_model(gp_models, scaler_X, scalers, original_y, scaler_interaction, scaler_poly, poly, n_iterations=1500, K=50, lr=0.0003):
    """
    Train MLP with increased iterations and hidden dimensions.
    """
    logger.info("Training Pareto Set Model...")
    model = ParetoSetModel(input_dim=2, output_dim=2, hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_min = torch.tensor([0.0, 0.0], dtype=torch.float32)
    x_max = torch.tensor([5.0, 15.0], dtype=torch.float32)
    
    target_region = {
        'min': np.array([205, 13, 3.5]),
        'max': np.array([225, 15, 5])
    }
    
    best_loss = float('inf')
    patience = 150
    counter = 0
    
    for t in range(n_iterations):
        optimizer.zero_grad()
        loss = 0.0
        
        n_per_group = K // 4
        lambda_max_strength = np.random.dirichlet([20, 1, 1], size=n_per_group)
        lambda_max_toughness = np.random.dirichlet([1, 20, 1], size=n_per_group)
        lambda_max_impact = np.random.dirichlet([1, 1, 20], size=n_per_group)
        lambda_balanced = np.array([0.33, 0.33, 0.34])[np.newaxis, :].repeat(K - 3*n_per_group, axis=0)
        lambda_vecs = np.vstack([lambda_max_strength, lambda_max_toughness, lambda_max_impact, lambda_balanced])
        lambda_vecs_2d = lambda_vecs[:, :2]
        lambda_tensor = torch.tensor(lambda_vecs_2d, dtype=torch.float32, requires_grad=True)
        
        x_pred = model(lambda_tensor)
        x_pred_scaled = x_pred * (x_max - x_min) + x_min
        
        interaction = (x_pred_scaled[:, 0] * x_pred_scaled[:, 1]).unsqueeze(1)
        interaction_np = interaction.detach().numpy()
        interaction_scaled = scaler_interaction.transform(interaction_np)
        x_poly = poly.transform(x_pred_scaled.detach().numpy())
        x_poly_scaled = scaler_poly.transform(x_poly)
        x_full = torch.cat((torch.tensor(x_poly_scaled, dtype=torch.float32), torch.tensor(interaction_scaled, dtype=torch.float32)), dim=1)
        x_scaled = scaler_X.transform(x_full.detach().numpy())
        x_scaled_np = x_scaled
        
        f_pred, _ = predict_gp(gp_models, x_scaled_np, scalers, original_y)
        
        z_star = np.max(f_pred, axis=0)
        
        for k in range(K):
            g_tch = augmented_tch_scalarization(f_pred[k], lambda_vecs[k], z_star, target_region=target_region)
            loss += g_tch
        
        loss /= K
        loss_tensor = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()
        
        if t % 100 == 0:
            logger.info(f"Iteration {t}, Loss: {loss:.4f}")
        
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            logger.info(f"Early stopping at iteration {t}")
            break
    
    return model

# Step 8: Evaluate Pareto Front with Targeted λ Sampling
def evaluate_pareto_front(model, gp_models, scaler_X, scalers, original_y, scaler_interaction, scaler_poly, poly, n_vectors=1000):
    """
    Evaluate 1000 preference vectors with composition constraints.
    """
    logger.info("Evaluating Pareto front...")
    x_min = torch.tensor([0.0, 0.0], dtype=torch.float32)
    x_max = torch.tensor([5.0, 15.0], dtype=torch.float32)
    
    n_per_group = n_vectors // 4
    lambda_max_strength = np.random.dirichlet([20, 1, 1], size=n_per_group)
    lambda_max_toughness = np.random.dirichlet([1, 20, 1], size=n_per_group)
    lambda_max_impact = np.random.dirichlet([1, 1, 20], size=n_per_group)
    lambda_balanced = np.array([0.33, 0.33, 0.34])[np.newaxis, :].repeat(n_vectors - 3*n_per_group, axis=0)
    lambda_vecs = np.vstack([lambda_max_strength, lambda_max_toughness, lambda_max_impact, lambda_balanced])
    lambda_vecs_2d = lambda_vecs[:, :2]
    lambda_tensor = torch.tensor(lambda_vecs_2d, dtype=torch.float32)
    
    with torch.no_grad():
        x_pred = model(lambda_tensor)
        x_pred_scaled = x_pred * (x_max - x_min) + x_min
        interaction = (x_pred_scaled[:, 0] * x_pred_scaled[:, 1]).unsqueeze(1)
        interaction_np = interaction.numpy()
        interaction_scaled = scaler_interaction.transform(interaction_np)
        x_poly = poly.transform(x_pred_scaled.numpy())
        x_poly_scaled = scaler_poly.transform(x_poly)
        x_full = torch.cat((torch.tensor(x_poly_scaled, dtype=torch.float32), torch.tensor(interaction_scaled, dtype=torch.float32)), dim=1)
        x_scaled = scaler_X.transform(x_full.numpy())
        f_pred, f_std = predict_gp(gp_models, x_scaled, scalers, original_y)
        pok = 100 - x_pred_scaled[:, 0] - x_pred_scaled[:, 1]
        compositions = np.column_stack((x_pred_scaled[:, 0], x_pred_scaled[:, 1], pok))
        
        valid = (compositions[:, 0] + compositions[:, 1] <= 20)
        compositions = compositions[valid]
        f_pred = f_pred[valid]
        f_std = f_std[valid]
        lambda_vecs = lambda_vecs[valid]
        if len(compositions) < n_vectors:
            logger.warning(f"Only {len(compositions)} valid solutions after composition constraint. Padding with random samples.")
            extra_indices = np.random.choice(len(compositions), size=n_vectors - len(compositions), replace=True)
            compositions = np.vstack([compositions, compositions[extra_indices]])
            f_pred = np.vstack([f_pred, f_pred[extra_indices]])
            f_std = np.vstack([f_std, f_std[extra_indices]])
            lambda_vecs = np.vstack([lambda_vecs, lambda_vecs[extra_indices]])
    
    return compositions, f_pred, f_std, lambda_vecs

# Step 9: Analyze Trade-Offs
def analyze_tradeoffs(F_pareto, lambda_vecs, objectives):
    """
    Compute trade-off metrics and target coverage.
    """
    logger.info("Analyzing trade-offs...")
    corr_matrix = np.corrcoef(F_pareto.T)
    max_indices = np.argmax(F_pareto, axis=0)
    extreme_solutions = []
    for i, obj in enumerate(objectives):
        max_solution = {
            'Objective': obj,
            'Max Value': F_pareto[max_indices[i], i],
            'At Strength': F_pareto[max_indices[i], 0],
            'At Fracture Toughness': F_pareto[max_indices[i], 1],
            'At Impact Energy': F_pareto[max_indices[i], 2],
            'Weight Strength': lambda_vecs[max_indices[i], 0],
            'Weight Fracture Toughness': lambda_vecs[max_indices[i], 1],
            'Weight Impact Energy': lambda_vecs[max_indices[i], 2]
        }
        extreme_solutions.append(max_solution)
    
    target_center = np.array([215, 14, 4.25])
    distances = np.linalg.norm(F_pareto - target_center, axis=1)
    balanced_idx = np.argmin(distances)
    balanced_solution = {
        'Objective': 'Balanced',
        'Max Value': distances[balanced_idx],
        'At Strength': F_pareto[balanced_idx, 0],
        'At Fracture Toughness': F_pareto[balanced_idx, 1],
        'At Impact Energy': F_pareto[balanced_idx, 2],
        'Weight Strength': lambda_vecs[balanced_idx, 0],
        'Weight Fracture Toughness': lambda_vecs[balanced_idx, 1],
        'Weight Impact Energy': lambda_vecs[balanced_idx, 2]
    }
    extreme_solutions.append(balanced_solution)
    
    max_values = np.max(F_pareto, axis=0)
    min_values = np.min(F_pareto, axis=0)
    normalized_F = (F_pareto - min_values) / (max_values - min_values + 1e-6)
    losses = []
    for i in range(len(F_pareto)):
        max_obj_idx = np.argmax(normalized_F[i])
        loss = {
            'Solution': i,
            'Max Objective': objectives[max_obj_idx],
            'Strength (%)': (max_values[0] - F_pareto[i, 0]) / max_values[0] * 100,
            'Fracture Toughness (%)': (max_values[1] - F_pareto[i, 1]) / max_values[1] * 100,
            'Impact Energy (%)': (max_values[2] - F_pareto[i, 2]) / max_values[2] * 100,
            'Weight Strength': lambda_vecs[i, 0],
            'Weight Fracture Toughness': lambda_vecs[i, 1],
            'Weight Impact Energy': lambda_vecs[i, 2]
        }
        losses.append(loss)
    
    # Calculate target coverage
    target_coverage = np.sum(
        (F_pareto[:, 0] >= 205) & (F_pareto[:, 0] <= 225) &
        (F_pareto[:, 1] >= 13) & (F_pareto[:, 1] <= 15) &
        (F_pareto[:, 2] >= 3.5) & (F_pareto[:, 2] <= 5)
    ) / len(F_pareto) * 100
    
    return corr_matrix, pd.DataFrame(extreme_solutions), pd.DataFrame(losses), target_coverage

# Step 10: Visualize Pareto Front
def visualize_pareto_front(compositions, F_pareto, F_std, lambda_vecs, output_dir, filename_prefix="pareto_initial"):
    """
    Visualize objective space and composition distribution.
    """
    logger.info(f"Visualizing Pareto front: {filename_prefix}")
    objectives = ['Strength (MPa)', 'Fracture Toughness (MPa·m^1/2)', 'Impact Energy Dissipation (J)']
    components = ['TPU (%)', 'BF (%)', 'POK (%)']
    
    corr_matrix, df_extremes, df_losses, target_coverage = analyze_tradeoffs(F_pareto, lambda_vecs, objectives)
    print("\nObjective Correlation Matrix:")
    print(pd.DataFrame(corr_matrix, index=objectives, columns=objectives).round(4))
    print(f"\nTarget Coverage (% in 205–225 MPa, 13–15 MPa·m^1/2, 3.5–5 J): {target_coverage:.2f}%")
    print("\nExtreme and Balanced Solutions:")
    print(df_extremes.round(4))
    print("\nNormalized Losses (Top 10, % drop from max):")
    print(df_losses.head(10).round(4))
    df_losses.to_csv(os.path.join(output_dir, f"{filename_prefix}_tradeoffs.csv"), index=False, encoding='utf-8')
    
    max_strength_idx = np.argmax(F_pareto[:, 0])
    max_toughness_idx = np.argmax(F_pareto[:, 1])
    max_impact_idx = np.argmax(F_pareto[:, 2])
    target_center = np.array([215, 14, 4.25])
    balanced_idx = np.argmin(np.linalg.norm(F_pareto - target_center, axis=1))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(F_pareto[:, 0], F_pareto[:, 1], F_pareto[:, 2], c=lambda_vecs[:, 0], cmap='viridis', label='Pareto Front')
    ax.scatter([205, 225], [13, 15], [3.5, 5], c='red', s=100, marker='s', label='Target Region')
    ax.scatter(F_pareto[max_strength_idx, 0], F_pareto[max_strength_idx, 1], F_pareto[max_strength_idx, 2], c='purple', s=200, marker='*', label='Max Strength')
    ax.scatter(F_pareto[max_toughness_idx, 0], F_pareto[max_toughness_idx, 1], F_pareto[max_toughness_idx, 2], c='orange', s=200, marker='*', label='Max Toughness')
    ax.scatter(F_pareto[max_impact_idx, 0], F_pareto[max_impact_idx, 1], F_pareto[max_impact_idx, 2], c='green', s=200, marker='*', label='Max Impact')
    ax.scatter(F_pareto[balanced_idx, 0], F_pareto[balanced_idx, 1], F_pareto[balanced_idx, 2], c='blue', s=200, marker='*', label='Balanced')
    fig.colorbar(scatter, label='Weight for Strength')
    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.set_zlabel(objectives[2])
    ax.set_title(f'Pareto Front ({len(F_pareto)} Solutions)')
    ax.legend()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_objectives_3d.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        scatter = ax.scatter(F_pareto[:, i], F_pareto[:, j], c=lambda_vecs[:, 0], cmap='viridis', label='Pareto Front')
        if i == 0 and j == 1:
            ax.plot([205, 205, 225, 225, 205], [13, 15, 15, 13, 13], 'r--', label='Target Region')
        elif i == 0 and j == 2:
            ax.plot([205, 205, 225, 225, 205], [3.5, 5, 5, 3.5, 3.5], 'r--', label='Target Region')
        elif i == 1 and j == 2:
            ax.plot([13, 13, 15, 15, 13], [3.5, 5, 5, 3.5, 3.5], 'r--', label='Target Region')
        ax.scatter(F_pareto[max_strength_idx, i], F_pareto[max_strength_idx, j], c='purple', s=200, marker='*', label='Max Strength')
        ax.scatter(F_pareto[max_toughness_idx, i], F_pareto[max_toughness_idx, j], c='orange', s=200, marker='*', label='Max Toughness')
        ax.scatter(F_pareto[max_impact_idx, i], F_pareto[max_impact_idx, j], c='green', s=200, marker='*', label='Max Impact')
        ax.scatter(F_pareto[balanced_idx, i], F_pareto[balanced_idx, j], c='blue', s=200, marker='*', label='Balanced')
        ax.set_xlabel(objectives[i])
        ax.set_ylabel(objectives[j])
        ax.set_title(f'{objectives[i]} vs {objectives[j]}')
        ax.legend()
        ax.grid(True)
    fig.colorbar(scatter, ax=axes, label='Weight for Strength')
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_objectives_2d.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(compositions[:, 0], compositions[:, 1], compositions[:, 2], c=lambda_vecs[:, 0], cmap='viridis', label='Composition Distribution')
    fig.colorbar(scatter, label='Weight for Strength')
    ax.set_xlabel(components[0])
    ax.set_ylabel(components[1])
    ax.set_zlabel(components[2])
    ax.set_title(f'Composition Distribution (TPU, BF, POK)')
    ax.legend()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_compositions_3d.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    comp_pairs = [(0, 1), (0, 2), (1, 2)]
    for idx, (i, j) in enumerate(comp_pairs):
        ax = axes[idx]
        scatter = ax.scatter(compositions[:, i], compositions[:, j], c=lambda_vecs[:, 0], cmap='viridis', label='Composition Distribution')
        ax.set_xlabel(components[i])
        ax.set_ylabel(components[j])
        ax.set_title(f'{components[i]} vs {components[j]}')
        ax.legend()
        ax.grid(True)
    fig.colorbar(scatter, ax=axes, label='Weight for Strength')
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_compositions_2d.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    df_pareto = pd.DataFrame({
        'TPU (%)': compositions[:, 0],
        'BF (%)': compositions[:, 1],
        'POK (%)': compositions[:, 2],
        'Predicted Strength (MPa)': F_pareto[:, 0],
        'Strength Std (MPa)': F_std[:, 0],
        'Predicted Fracture Toughness (MPa·m^1/2)': F_pareto[:, 1],
        'Fracture Toughness Std (MPa·m^1/2)': F_std[:, 1],
        'Predicted Impact Energy Dissipation (J)': F_pareto[:, 2],
        'Impact Energy Std (J)': F_std[:, 2],
        'Weight Strength': lambda_vecs[:, 0],
        'Weight Fracture Toughness': lambda_vecs[:, 1],
        'Weight Impact Energy': lambda_vecs[:, 2]
    })
    df_pareto.to_csv(os.path.join(output_dir, f"{filename_prefix}_solutions.csv"), index=False, encoding='utf-8')
    print("\nPareto Optimal Solutions (Top 10):")
    print(df_pareto.head(10).round(4))
    
    # Log Std and relative Std statistics
    std_to_pred_mapping = {
        'Strength Std (MPa)': 'Predicted Strength (MPa)',
        'Fracture Toughness Std (MPa·m^1/2)': 'Predicted Fracture Toughness (MPa·m^1/2)',
        'Impact Energy Std (J)': 'Predicted Impact Energy Dissipation (J)'
    }
    for metric, col, target_mean, target_min, target_max, rel_mean, rel_max in [
        ('Strength', 'Strength Std (MPa)', 4.75, 2.94, 13.79, 2.75, 7.98),
        ('Fracture Toughness', 'Fracture Toughness Std (MPa·m^1/2)', 0.70, 0.51, 1.60, 7.26, 16.58),
        ('Impact Energy', 'Impact Energy Std (J)', 0.17, 0.12, 0.38, 5.13, 11.29)
    ]:
        std_values = df_pareto[col].values
        pred_col = std_to_pred_mapping[col]
        pred_values = df_pareto[pred_col].values
        mean_std = np.mean(std_values)
        min_std = np.min(std_values)
        max_std = np.max(std_values)
        mean_pred = np.mean(pred_values)
        rel_std_mean = (mean_std / mean_pred * 100) if mean_pred != 0 else 0
        rel_std_max = (max_std / mean_pred * 100) if mean_pred != 0 else 0
        logger.info(f"{metric}: Mean Std = {mean_std:.2f}, Min Std = {min_std:.2f}, Max Std = {max_std:.2f}, Mean Pred = {mean_pred:.2f}, Rel Std Mean = {rel_std_mean:.2f}%, Rel Std Max = {rel_std_max:.2f}%")
        if mean_std > target_mean * 1.1 or mean_std < target_mean * 0.9:
            logger.warning(f"{metric} Mean Std ({mean_std:.2f}) outside target range [{target_mean*0.9:.2f}, {target_mean*1.1:.2f}]")
        if min_std < target_min * 0.9:
            logger.warning(f"{metric} Min Std ({min_std:.2f}) below target ({target_min:.2f})")
        if max_std > target_max * 1.1:
            logger.warning(f"{metric} Max Std ({max_std:.2f}) exceeds target ({target_max:.2f})")
        if rel_std_mean > rel_mean * 1.1 or rel_std_mean < rel_mean * 0.9:
            logger.warning(f"{metric} Rel Std Mean ({rel_std_mean:.2f}%) outside target range [{rel_mean*0.9:.2f}, {rel_mean*1.1:.2f}]%")
        if rel_std_max > rel_max * 1.1:
            logger.warning(f"{metric} Rel Std Max ({rel_std_max:.2f}%) exceeds target ({rel_max:.2f})%")

def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    X, y, indices, scaler_interaction, scaler_poly, poly, compositions_orig = load_data(args.data_file, remove_outliers=True)
    X_scaled, y_scaled, scaler_X, scalers = preprocess_data(X, y, output_dir)
    
    print("\n=== Initial 5-fold Cross-Validation (GP) ===")
    avg_mse, avg_r2, avg_coverage, _ = cross_validate_gp(X_scaled, y_scaled, scalers, y, indices, output_dir, n_splits=5)
    
    print("\n=== Generating Pareto Front (1000 Preference Vectors) ===")
    gp_models = load_gp_models(output_dir)
    pareto_set_model = train_pareto_set_model(gp_models, scaler_X, scalers, y, scaler_interaction, scaler_poly, poly, n_iterations=1500, K=50)
    compositions, f_pareto, f_std, lambda_vecs = evaluate_pareto_front(pareto_set_model, gp_models, scaler_X, scalers, y, scaler_interaction, scaler_poly, poly, n_vectors=1000)
    visualize_pareto_front(compositions, f_pareto, f_std, lambda_vecs, output_dir, filename_prefix="pareto_initial")
    
    logger.info(f"Pareto front with {len(compositions)} valid solutions saved to pareto_initial_solutions.csv")
    logger.info(f"POK range: {compositions[:, 2].min():.2f} to {compositions[:, 2].max():.2f}")
    logger.info(f"Strength range: {f_pareto[:, 0].min():.2f} to {f_pareto[:, 0].max():.2f}")
    logger.info(f"Strength Std range: {f_std[:, 0].min():.2f} to {f_std[:, 0].max():.2f}")
    logger.info("GP models and scalers loaded and used successfully")

if __name__ == "__main__":
    main()