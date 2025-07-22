import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import argparse
from typing import Tuple, List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for data file and output directory."""
    parser = argparse.ArgumentParser(description='Pareto Set Learning with Active Learning')
    parser.add_argument('--data_file', type=str, default='/data/input_data.csv',
                        help='Path to input CSV file with data')
    parser.add_argument('--output_dir', type=str, default='/results',
                        help='Directory to save output files')
    return parser.parse_args()

def load_data(data_file: str, remove_outliers: bool = True, outlier_indices: Optional[List[int]] = None, encoding: str = 'utf-8') -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler, PolynomialFeatures, np.ndarray]:
    """
    Load data from CSV, remove specified outliers, and preprocess features.
    
    Args:
        data_file: Path to CSV file with columns ['TPU (%)', 'BF (%)', 'Strength (MPa)', 'Fracture Toughness (MPa·m1/2)', 'Impact Energy Dissipation (J)']
        remove_outliers: Whether to remove outliers
        outlier_indices: List of indices to remove if remove_outliers is True
        encoding: File encoding for CSV (default: 'utf-8')
    
    Returns:
        Tuple of (X, y, indices, scaler_interaction, scaler_poly, poly, compositions)
    
    Raises:
        FileNotFoundError: If the data file does not exist
        ValueError: If required columns are missing
        UnicodeDecodeError: If the file encoding is incorrect
    """
    logger.info(f"Loading data from {data_file} with encoding {encoding}...")
    
    if not os.path.exists(data_file):
        logger.error(f"Data file {data_file} not found.")
        raise FileNotFoundError(f"Data file {data_file} not found.")
    
    try:
        df = pd.read_csv(data_file, encoding=encoding)
    except UnicodeDecodeError:
        logger.warning(f"Failed to read {data_file} with {encoding} encoding. Trying 'utf-8-sig'...")
        try:
            df = pd.read_csv(data_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            logger.warning(f"Failed to read {data_file} with utf-8-sig encoding. Trying 'cp1252'...")
            try:
                df = pd.read_csv(data_file, encoding='cp1252')
            except UnicodeDecodeError as e:
                logger.error(f"Cannot decode {data_file}. Tried UTF-8, UTF-8-SIG, and CP1252. Please check file encoding.")
                raise e
    
    # Debug: Print column names
    logger.info(f"Columns in CSV: {df.columns.tolist()}")
    
    required_columns = ['TPU (%)', 'BF (%)', 'Strength (MPa)', 'Fracture Toughness (MPa·m1/2)', 'Impact Energy Dissipation (J)']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Data file must contain columns: {required_columns}")
        raise ValueError(f"Data file must contain columns: {required_columns}")
    
    if remove_outliers and outlier_indices is not None:
        valid_outliers = [i for i in outlier_indices if i in df.index]
        if valid_outliers:
            logger.info(f"Removing outliers: {valid_outliers}")
            df = df.drop(index=valid_outliers)
        else:
            logger.warning("No valid outlier indices found in the data.")
    
    X = df[['TPU (%)', 'BF (%)']].values
    interaction = (df['TPU (%)'] * df['BF (%)']).values.reshape(-1, 1)
    scaler_interaction = StandardScaler()
    X_interaction = scaler_interaction.fit_transform(interaction)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)
    scaler_poly = StandardScaler()
    X_poly_scaled = scaler_poly.fit_transform(X_poly)
    X = np.column_stack((X_poly_scaled, X_interaction))
    y = df[['Strength (MPa)', 'Fracture Toughness (MPa·m1/2)', 'Impact Energy Dissipation (J)']].values
    df['POK (%)'] = 100 - df['TPU (%)'] - df['BF (%)']
    return X, y, df.index.values, scaler_interaction, scaler_poly, poly, df[['TPU (%)', 'BF (%)', 'POK (%)']].values

def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Scale inputs and log-transform outputs."""
    logger.info("Preprocessing data...")
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_log = np.log(y + 1e-1) * np.sign(y + 1e-1)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_log)
    return X_scaled, y_scaled, scaler_X, scaler_y

def inverse_transform_y(y_pred_scaled: np.ndarray, scaler_y: StandardScaler, original_y: np.ndarray) -> np.ndarray:
    """Inverse transform scaled and log-transformed outputs."""
    y_log = scaler_y.inverse_transform(y_pred_scaled)
    y_unscaled = np.exp(y_log * np.sign(y_log)) - 1e-1
    return y_unscaled

def train_gp_models(X_train: np.ndarray, y_train: np.ndarray) -> List[GaussianProcessRegressor]:
    """Train GP models with Matern kernel (ν=2.5)."""
    logger.info("Training GP models...")
    matern_kernel = Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-3, 1e3)) + \
                    WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-7, 1e1))
    
    gp_models = []
    for i in range(y_train.shape[1]):
        gp_matern = GaussianProcessRegressor(kernel=matern_kernel, n_restarts_optimizer=100, random_state=42)
        try:
            gp_matern.fit(X_train, y_train[:, i])
            gp_models.append(gp_matern)
            logger.info(f"GP Model {i+1} Matern Kernel: {gp_matern.kernel_}")
        except Exception as e:
            logger.error(f"GP Model {i+1} failed: {e}")
            raise
    return gp_models

def predict_gp(gp_models: List[GaussianProcessRegressor], X: np.ndarray, scaler_y: StandardScaler, original_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with GP models."""
    logger.info("Predicting with GP models...")
    means, stds = [], []
    for gp_matern in gp_models:
        mean, std = gp_matern.predict(X, return_std=True)
        means.append(mean)
        stds.append(std)
    y_pred_scaled = np.column_stack(means)
    y_std_scaled = np.column_stack(stds)
    y_pred = inverse_transform_y(y_pred_scaled, scaler_y, original_y)
    y_std = y_std_scaled * np.sqrt(scaler_y.var_) * np.exp(np.abs(scaler_y.mean_))
    return y_pred, y_std

def input_real_values(selected_compositions: np.ndarray, selected_f_pred: np.ndarray, selected_f_std: np.ndarray, round_idx: int, output_dir: str) -> np.ndarray:
    """
    Require user to input real experimental values and save them.
    
    Args:
        selected_compositions: Selected compositions (TPU, BF, POK)
        selected_f_pred: Predicted objective values
        selected_f_std: Prediction uncertainties
        round_idx: Current round index
        output_dir: Directory to save output files
    
    Returns:
        Array of real experimental values
    """
    logger.info(f"Processing real values for Round {round_idx}...")
    df_selected = pd.DataFrame({
        'TPU (%)': selected_compositions[:, 0],
        'BF (%)': selected_compositions[:, 1],
        'POK (%)': selected_compositions[:, 2],
        'Predicted Strength (MPa)': selected_f_pred[:, 0],
        'Strength Std (MPa)': selected_f_std[:, 0],
        'Predicted Fracture Toughness (MPa·m1/2)': selected_f_pred[:, 1],
        'Fracture Toughness Std (MPa·m1/2)': selected_f_std[:, 1],
        'Predicted Impact Energy Dissipation (J)': selected_f_pred[:, 2],
        'Impact Energy Std (J)': selected_f_std[:, 2]
    })
    selected_file = os.path.join(output_dir, f"al_round_{round_idx}_selected.csv")
    df_selected.to_csv(selected_file, index=False)
    print(f"\nRound {round_idx} Selected Points for Experimental Validation (saved to {selected_file}):")
    print(df_selected.round(4))
    
    print(f"\nPlease provide real experimental values for Round {round_idx} in a CSV file with columns:")
    print("['Real Strength (MPa)', 'Real Fracture Toughness (MPa·m1/2)', 'Real Impact Energy Dissipation (J)']")
    real_data_file = input(f"Enter the path to the real data CSV for Round {round_idx}: ")
    
    try:
        real_data = pd.read_csv(real_data_file, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning(f"Failed to read {real_data_file} with utf-8 encoding. Trying 'utf-8-sig'...")
        try:
            real_data = pd.read_csv(real_data_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            logger.warning(f"Failed to read {real_data_file} with utf-8-sig encoding. Trying 'cp1252'...")
            try:
                real_data = pd.read_csv(real_data_file, encoding='cp1252')
            except UnicodeDecodeError as e:
                logger.error(f"Cannot decode {real_data_file}. Tried UTF-8, UTF-8-SIG, and CP1252. Please check file encoding.")
                raise e
    
    required_columns = ['Real Strength (MPa)', 'Real Fracture Toughness (MPa·m1/2)', 'Real Impact Energy Dissipation (J)']
    if not all(col in real_data.columns for col in required_columns):
        logger.error(f"Real data CSV must contain columns: {required_columns}")
        raise ValueError(f"Real data CSV must contain columns: {required_columns}")
    if len(real_data) != len(selected_compositions):
        logger.error(f"Real data CSV must have {len(selected_compositions)} rows, matching selected compositions")
        raise ValueError(f"Real data CSV must have {len(selected_compositions)} rows, matching selected compositions")
    
    y_real = real_data[required_columns].values
    df_real = pd.DataFrame({
        'TPU (%)': selected_compositions[:, 0],
        'BF (%)': selected_compositions[:, 1],
        'POK (%)': selected_compositions[:, 2],
        'Real Strength (MPa)': y_real[:, 0],
        'Real Fracture Toughness (MPa·m1/2)': y_real[:, 1],
        'Real Impact Energy Dissipation (J)': y_real[:, 2]
    })
    real_file = os.path.join(output_dir, f"al_round_{round_idx}_real.csv")
    df_real.to_csv(real_file, index=False)
    print(f"\nRound {round_idx} Real Experimental Values (saved to {real_file}):")
    print(df_real.round(4))
    return y_real

def cross_validate_gp(X_scaled: np.ndarray, y_scaled: np.ndarray, scaler_y: StandardScaler, original_y: np.ndarray, indices: np.ndarray, n_splits: int = 5, output_dir: str = 'results') -> Tuple[Dict, Dict, Dict, Dict]:
    """Perform k-fold cross-validation with outlier detection."""
    logger.info("Performing cross-validation...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = {0: [], 1: [], 2: []}
    r2_scores = {0: [], 1: [], 2: []}
    coverage_scores = {0: [], 1: [], 2: []}
    residuals = {0: [], 1: [], 2: []}
    outlier_indices = {0: [], 1: [], 2: []}
    objectives = ['Strength (MPa)', 'Fracture Toughness (MPa·m1/2)', 'Impact Energy Dissipation (J)']
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        logger.info(f"Processing fold {fold_idx+1}/{n_splits}")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
        y_test_orig = original_y[test_idx]
        
        gp_models = train_gp_models(X_train, y_train)
        y_pred, y_std = predict_gp(gp_models, X_test, scaler_y, original_y)
        
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
    
    avg_mse = {i: np.mean(mse_scores[i]) for i in range(3)}
    avg_r2 = {i: np.mean(r2_scores[i]) for i in range(3)}
    avg_coverage = {i: np.mean(coverage_scores[i]) for i in range(3)}
    max_residuals = {i: np.max(np.abs(np.concatenate(residuals[i]))) for i in range(3)}
    
    # Save cross-validation results
    cv_results = pd.DataFrame({
        'Objective': objectives,
        'Average MSE': [avg_mse[i] for i in range(3)],
        'Average R²': [avg_r2[i] for i in range(3)],
        'Average 95% CI Coverage': [avg_coverage[i] for i in range(3)],
        'Max Absolute Residual': [max_residuals[i] for i in range(3)],
        'Outliers': [list(set(outlier_indices[i])) for i in range(3)]
    })
    cv_file = os.path.join(output_dir, 'cross_validation_results.csv')
    cv_results.to_csv(cv_file, index=False)
    print(f"\nCross-Validation Results (saved to {cv_file}):")
    print(cv_results.round(4))
    
    return avg_mse, avg_r2, avg_coverage, residuals

class ParetoSetModel(nn.Module):
    """MLP to map preference vectors to Pareto solutions."""
    def __init__(self, input_dim: int = 2, output_dim: int = 2, hidden_dim: int = 128):
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

def augmented_tch_scalarization(f_pred: np.ndarray, lambda_vec: np.ndarray, z_star: np.ndarray, rho: float = 0.01, epsilon_factor: float = 0.05, target_region: Optional[Dict] = None) -> float:
    """Compute augmented TCH scalarization with strong penalty."""
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

def train_pareto_set_model(gp_models: List[GaussianProcessRegressor], scaler_X: StandardScaler, scaler_y: StandardScaler, original_y: np.ndarray, 
                          scaler_interaction: StandardScaler, scaler_poly: StandardScaler, poly: PolynomialFeatures, 
                          n_iterations: int = 1500, K: int = 50, lr: float = 0.0003, 
                          x_min: np.ndarray = np.array([0.0, 0.0]), x_max: np.ndarray = np.array([5.0, 15.0]), 
                          target_region: Optional[Dict] = None) -> ParetoSetModel:
    """Train MLP with increased iterations and hidden dimensions."""
    logger.info("Training Pareto Set Model...")
    model = ParetoSetModel(input_dim=2, output_dim=2, hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_min = torch.tensor(x_min, dtype=torch.float32)
    x_max = torch.tensor(x_max, dtype=torch.float32)
    
    if target_region is None:
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
        
        f_pred, _ = predict_gp(gp_models, x_scaled, scaler_y, original_y)
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

def compute_hypervolume(y: np.ndarray, reference_point: np.ndarray) -> float:
    """Compute hypervolume of a set of solutions (maximization)."""
    logger.info("Computing hypervolume...")
    y_neg = -y
    ref_neg = -reference_point
    sorted_y = np.sort(y_neg, axis=0)
    hv = 0.0
    for i in range(len(sorted_y)):
        if i == 0:
            hv += np.prod(ref_neg - sorted_y[i])
        else:
            hv += np.prod(ref_neg - sorted_y[i]) - np.prod(ref_neg - sorted_y[i-1])
    return hv

def evaluate_pareto_front(model: ParetoSetModel, gp_models: List[GaussianProcessRegressor], scaler_X: StandardScaler, scaler_y: StandardScaler, 
                         original_y: np.ndarray, scaler_interaction: StandardScaler, scaler_poly: StandardScaler, poly: PolynomialFeatures, 
                         n_vectors: int = 1000, x_min: np.ndarray = np.array([0.0, 0.0]), x_max: np.ndarray = np.array([5.0, 15.0]), 
                         output_dir: str = 'results') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Pareto front and save results."""
    logger.info("Evaluating Pareto front...")
    x_min = torch.tensor(x_min, dtype=torch.float32)
    x_max = torch.tensor(x_max, dtype=torch.float32)
    
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
        f_pred, f_std = predict_gp(gp_models, x_scaled, scaler_y, original_y)
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
    
    # Save Pareto front data
    df_pareto = pd.DataFrame({
        'TPU (%)': compositions[:, 0],
        'BF (%)': compositions[:, 1],
        'POK (%)': compositions[:, 2],
        'Predicted Strength (MPa)': f_pred[:, 0],
        'Strength Std (MPa)': f_std[:, 0],
        'Predicted Fracture Toughness (MPa·m1/2)': f_pred[:, 1],
        'Fracture Toughness Std (MPa·m1/2)': f_std[:, 1],
        'Predicted Impact Energy Dissipation (J)': f_pred[:, 2],
        'Impact Energy Std (J)': f_std[:, 2],
        'Weight Strength': lambda_vecs[:, 0],
        'Weight Fracture Toughness': lambda_vecs[:, 1],
        'Weight Impact Energy': lambda_vecs[:, 2]
    })
    pareto_file = os.path.join(output_dir, 'pareto_solutions.csv')
    df_pareto.to_csv(pareto_file, index=False)
    print(f"\nPareto Optimal Solutions (Top 10, saved to {pareto_file}):")
    print(df_pareto.head(10).round(4))
    
    return compositions, f_pred, f_std, lambda_vecs

def analyze_tradeoffs(F_pareto: np.ndarray, lambda_vecs: np.ndarray, objectives: List[str], output_dir: str) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Compute trade-off metrics and save results."""
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
    
    # Save trade-off results
    corr_df = pd.DataFrame(corr_matrix, index=objectives, columns=objectives)
    corr_file = os.path.join(output_dir, 'correlation_matrix.csv')
    corr_df.to_csv(corr_file)
    print(f"\nObjective Correlation Matrix (saved to {corr_file}):")
    print(corr_df.round(4))
    
    extremes_df = pd.DataFrame(extreme_solutions)
    extremes_file = os.path.join(output_dir, 'extreme_solutions.csv')
    extremes_df.to_csv(extremes_file, index=False)
    print(f"\nExtreme and Balanced Solutions (saved to {extremes_file}):")
    print(extremes_df.round(4))
    
    losses_df = pd.DataFrame(losses)
    losses_file = os.path.join(output_dir, 'tradeoff_losses.csv')
    losses_df.to_csv(losses_file, index=False)
    print(f"\nNormalized Losses (Top 10, % drop from max, saved to {losses_file}):")
    print(losses_df.head(10).round(4))
    
    return corr_matrix, extremes_df, losses_df

def active_learning(gp_models: List[GaussianProcessRegressor], scaler_X: StandardScaler, scaler_y: StandardScaler, original_y: np.ndarray, 
                   scaler_interaction: StandardScaler, scaler_poly: StandardScaler, poly: PolynomialFeatures, 
                   X_scaled: np.ndarray, y_scaled: np.ndarray, compositions: np.ndarray, output_dir: str, 
                   beta: float = 0.4, max_rounds: int = 4, batch_size: int = 5, n_vectors: int = 1000, 
                   x_min: np.ndarray = np.array([0.0, 0.0]), x_max: np.ndarray = np.array([5.0, 15.0])) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[GaussianProcessRegressor]]:
    """Perform active learning until HVI gain is less than 1% or max_rounds is reached."""
    logger.info("Starting Active Learning...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    current_X_scaled = X_scaled.copy()
    current_y_scaled = y_scaled.copy()
    current_y = original_y.copy()
    current_compositions = compositions.copy()
    reference_point = np.array([250, 16, 6])
    
    al_results = []
    hvi_values = []
    round_idx = 0
    
    while round_idx < max_rounds:
        round_idx += 1
        print(f"\nActive Learning Round {round_idx}")
        
        pareto_set_model = train_pareto_set_model(gp_models, scaler_X, scaler_y, current_y, scaler_interaction, scaler_poly, poly, 
                                                 n_iterations=1500, K=50, x_min=x_min, x_max=x_max)
        comp_candidates, f_pred, f_std, lambda_vecs = evaluate_pareto_front(pareto_set_model, gp_models, scaler_X, scaler_y, current_y, 
                                                                          scaler_interaction, scaler_poly, poly, n_vectors=n_vectors, 
                                                                          x_min=x_min, x_max=x_max, output_dir=output_dir)
        
        lcb = f_pred - beta * f_std
        
        selected_indices = []
        current_hv = compute_hypervolume(current_y, reference_point)
        selected_comps = []
        for _ in range(batch_size):
            best_hvi = -np.inf
            best_idx = -1
            for i in range(len(f_pred)):
                if i in selected_indices:
                    continue
                comp = comp_candidates[i, :2]
                if selected_comps and np.min([np.linalg.norm(comp - sc) for sc in selected_comps]) < 1.5:
                    continue
                temp_y = np.vstack([current_y, lcb[i]])
                hvi = compute_hypervolume(temp_y, reference_point) - current_hv
                if hvi > best_hvi:
                    best_hvi = hvi
                    best_idx = i
            if best_idx >= 0:
                selected_indices.append(best_idx)
                selected_comps.append(comp_candidates[best_idx, :2])
        
        if len(selected_indices) < batch_size:
            print(f"Warning: Only {len(selected_indices)} points selected in Round {round_idx}")
            additional_indices = np.random.choice([i for i in range(len(f_pred)) if i not in selected_indices], 
                                               size=batch_size - len(selected_indices), replace=False)
            selected_indices.extend(additional_indices)
        
        selected_compositions = comp_candidates[selected_indices]
        selected_f_pred = f_pred[selected_indices]
        selected_f_std = f_std[selected_indices]
        
        selected_y_real = input_real_values(selected_compositions, selected_f_pred, selected_f_std, round_idx, output_dir)
        
        selected_X_base = selected_compositions[:, :2]
        selected_X_poly = scaler_poly.transform(poly.transform(selected_X_base))
        selected_X_interaction = scaler_interaction.transform((selected_compositions[:, 0] * selected_compositions[:, 1]).reshape(-1, 1))
        selected_X = np.column_stack((selected_X_poly, selected_X_interaction))
        selected_X_scaled = scaler_X.transform(selected_X)
        
        current_X_scaled = np.vstack([current_X_scaled, selected_X_scaled])
        current_y_scaled = np.vstack([current_y_scaled, scaler_y.transform(np.log(selected_y_real + 1e-1))])
        current_y = np.vstack([current_y, selected_y_real])
        current_compositions = np.vstack([current_compositions, selected_compositions])
        
        gp_models = train_gp_models(current_X_scaled, current_y_scaled)
        
        _, f_pareto, f_std, lambda_vecs = evaluate_pareto_front(pareto_set_model, gp_models, scaler_X, scaler_y, current_y, 
                                                              scaler_interaction, scaler_poly, poly, n_vectors=n_vectors, 
                                                              x_min=x_min, x_max=x_max, output_dir=output_dir)
        
        avg_mse, avg_r2, avg_coverage, _ = cross_validate_gp(current_X_scaled, current_y_scaled, scaler_y, current_y, 
                                                            np.arange(len(current_y)), n_splits=5, output_dir=output_dir)
        
        high_perf = np.sum((f_pareto[:, 0] >= 205) & (f_pareto[:, 0] <= 225) &
                          (f_pareto[:, 1] >= 13) & (f_pareto[:, 1] <= 15) &
                          (f_pareto[:, 2] >= 3.5) & (f_pareto[:, 2] <= 5)) / len(f_pareto) * 100
        
        hvi_values.append(current_hv)
        hvi_gain = 100.0 if round_idx == 1 else ((hvi_values[-1] - hvi_values[-2]) / hvi_values[-2] * 100)
        
        al_results.append({
            'round': round_idx,
            'selected_compositions': selected_compositions,
            'selected_predictions': selected_f_pred,
            'selected_real_values': selected_y_real,
            'selected_uncertainties': f_std,
            'avg_mse': avg_mse,
            'avg_r2': avg_r2,
            'avg_coverage': avg_coverage,
            'high_perf_percentage': high_perf,
            'pareto_front': f_pareto,
            'pareto_compositions': comp_candidates,
            'lambda_vecs': lambda_vecs,
            'hvi': current_hv,
            'hvi_gain': hvi_gain
        })
        
        print(f"\nRound {round_idx} Summary:")
        print(f"High-Performance Solutions (% in 205–225 MPa, 13–15 MPa·m1/2, 3.5–5 J): {high_perf:.2f}%")
        print(f"MSE: Strength={avg_mse[0]:.4f}, Fracture Toughness={avg_mse[1]:.4f}, Impact Energy={avg_mse[2]:.4f}")
        print(f"R²: Strength={avg_r2[0]:.4f}, Fracture Toughness={avg_r2[1]:.4f}, Impact Energy={avg_r2[2]:.4f}")
        print(f"Coverage: Strength={avg_coverage[0]:.4f}, Fracture Toughness={avg_coverage[1]:.4f}, Impact Energy={avg_coverage[2]:.4f}")
        print(f"HVI: {current_hv:.4f}, HVI Gain: {hvi_gain:.2f}%")
        
        # Save round summary
        summary_df = pd.DataFrame([{
            'Round': res['round'],
            'High_Performance_Percentage': res['high_perf_percentage'],
            'MSE_Strength': res['avg_mse'][0],
            'MSE_Fracture_Toughness': res['avg_mse'][1],
            'MSE_Impact_Energy': res['avg_mse'][2],
            'R2_Strength': res['avg_r2'][0],
            'R2_Fracture_Toughness': res['avg_r2'][1],
            'R2_Impact_Energy': res['avg_r2'][2],
            'Coverage_Strength': res['avg_coverage'][0],
            'Coverage_Fracture_Toughness': res['avg_coverage'][1],
            'Coverage_Impact_Energy': res['avg_coverage'][2],
            'HVI': res['hvi'],
            'HVI_Gain': res['hvi_gain']
        } for res in al_results])
        summary_file = os.path.join(output_dir, 'active_learning_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        # Check HVI gain for early stopping
        if round_idx > 1 and hvi_gain < 1.0:
            print(f"\nStopping Active Learning: HVI Gain ({hvi_gain:.2f}%) is less than 1%")
            break
    
    return al_results, current_X_scaled, current_y_scaled, current_y, current_compositions, gp_models

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define default parameters
    outlier_indices = [2, 6, 20, 22, 24, 25, 26, 28, 29]
    x_min = np.array([0.0, 0.0])
    x_max = np.array([5.0, 15.0])
    target_region = {
        'min': np.array([205, 13, 3.5]),
        'max': np.array([225, 15, 5])
    }
    
    X, y, indices, scaler_interaction, scaler_poly, poly, compositions_orig = load_data(args.data_file, remove_outliers=True, outlier_indices=outlier_indices, encoding='utf-8')
    X_scaled, y_scaled, scaler_X, scaler_y = preprocess_data(X, y)
    
    print("\nPerforming Initial 5-fold Cross-Validation (GP)...")
    avg_mse, avg_r2, avg_coverage, _ = cross_validate_gp(X_scaled, y_scaled, scaler_y, y, indices, n_splits=5, output_dir=args.output_dir)
    
    print("\nRunning Active Learning (Until HVI Gain < 1% or Max 4 Rounds, 1000 Preference Vectors)...")
    al_results, final_X_scaled, final_y_scaled, final_y, final_compositions, final_gp_models = active_learning(
        train_gp_models(X_scaled, y_scaled), scaler_X, scaler_y, y, scaler_interaction, scaler_poly, poly, 
        X_scaled, y_scaled, compositions_orig, args.output_dir, beta=0.8, max_rounds=4, batch_size=5, n_vectors=1000,
        x_min=x_min, x_max=x_max
    )
    
    print("\nFinal Cross-Validation After Active Learning...")
    final_mse, final_r2, final_coverage, _ = cross_validate_gp(final_X_scaled, final_y_scaled, scaler_y, final_y, 
                                                              np.arange(len(final_y)), n_splits=5, output_dir=args.output_dir)
    
    print("\nActive Learning Summary (saved to active_learning_summary.csv):")
    summary_df = pd.read_csv(os.path.join(args.output_dir, 'active_learning_summary.csv'))
    print(summary_df.round(4))
    
    print("\nGenerating Final Pareto Front...")
    final_pareto_set_model = train_pareto_set_model(final_gp_models, scaler_X, scaler_y, final_y, scaler_interaction, scaler_poly, poly, 
                                                  n_iterations=1500, K=50, x_min=x_min, x_max=x_max, target_region=target_region)
    final_compositions, final_f_pareto, final_f_std, final_lambda_vecs = evaluate_pareto_front(
        final_pareto_set_model, final_gp_models, scaler_X, scaler_y, final_y, scaler_interaction, scaler_poly, poly, 
        n_vectors=1000, x_min=x_min, x_max=x_max, output_dir=args.output_dir)
    
    objectives = ['Strength (MPa)', 'Fracture Toughness (MPa·m1/2)', 'Impact Energy Dissipation (J)']
    analyze_tradeoffs(final_f_pareto, final_lambda_vecs, objectives, args.output_dir)

if __name__ == "__main__":
    main()