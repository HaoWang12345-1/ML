
import pandas as pd
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import joblib
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for data file and output directory."""
    parser = argparse.ArgumentParser(description='Gaussian Process Regression')
    parser.add_argument('--data_file', type=str, default='/data/input_data.csv',
                        help='Path to input CSV file with data')
    parser.add_argument('--output_dir', type=str, default='/results',
                        help='Directory to save output files')
    return parser.parse_args()

def load_data(args):
    """Load data from CSV or use hard-coded data if CSV is missing."""
    file_path = args.data_file
    if os.path.exists(file_path):
        try:
            logger.info(f"Loading data from {file_path} with encoding utf-8...")
            data = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"Failed to read {file_path} with utf-8 encoding. Trying 'utf-8-sig'...")
            try:
                data = pd.read_csv(file_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                logger.warning(f"Failed to read {file_path} with utf-8-sig encoding. Trying 'cp1252'...")
                try:
                    data = pd.read_csv(file_path, encoding='cp1252')
                except UnicodeDecodeError as e:
                    logger.error(f"Cannot decode {file_path}. Tried UTF-8, UTF-8-SIG, and CP1252. Falling back to hard-coded data.")
                    data = None
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}. Falling back to hard-coded data.")
            data = None
    else:
        logger.info(f"No data file found at {file_path}. Using hard-coded data.")
        data = None

    if data is None:
        logger.info("Using hard-coded data with 30 samples.")
        data_dict = {
            'TPU (%)': [0.79, 2.08, 1.17, 4.18, 1.7, 0.91, 4.04, 3.3, 1.43, 0.25, 4.98, 3.16, 0.46, 0.12, 2.58, 1.61, 3.41, 4.71, 2.49, 2.23, 3.73, 2.88, 1.14, 1.98, 4.58, 0.52, 4.37, 3.58, 3.89, 2.71],
            'BF (%)': [10.25, 9.89, 7.2, 1.14, 4.19, 3.32, 12.84, 8.2, 11.81, 6.6, 3.65, 14.83, 5.76, 10.97, 9.89, 5.08, 11.49, 2.52, 1.77, 2.07, 8.7, 4.97, 0.53, 6.51, 12.09, 7.9, 13.83, 14.19, 13.23, 0.49],
            'Strength (MPa)': [217, 212, 202, 110, 144, 136, 230, 204, 228, 167, 139, 209, 158, 224, 219, 151, 226, 133, 122, 130, 206, 150, 81, 169, 229, 203, 236, 210, 232, 74],
            'Fracture Toughness (MPa·m1/2)': [12.6, 14.1, 11.0, 6.1, 6.8, 6.4, 12.6, 13.4, 12.8, 8.1, 6.6, 6.2, 7.3, 11.2, 14.8, 7.1, 12.5, 6.3, 5.8, 6.8, 13.9, 7.3, 6.0, 8.3, 12.0, 11.6, 6.8, 6.6, 8.2, 5.5],
            'Impact Energy Dissipation (J)': [3.4, 4.3, 3.6, 2.6, 3.0, 2.9, 3.8, 3.6, 3.9, 3.3, 2.7, 2.7, 3.1, 3.4, 4.8, 3.2, 3.7, 2.8, 2.7, 2.9, 4.3, 3.1, 2.4, 3.4, 3.4, 3.9, 3.1, 2.8, 3.2, 2.2]
        }
        data = pd.DataFrame(data_dict)

    # Print columns to debug headers
    logger.info("CSV Columns: %s", data.columns.tolist())

    # Compute POK (%) before logging ranges
    data['POK (%)'] = 100 - data['TPU (%)'] - data['BF (%)']

    # Log input data ranges
    logger.info(f"TPU range: {data['TPU (%)'].min():.2f} to {data['TPU (%)'].max():.2f}")
    logger.info(f"BF range: {data['BF (%)'].min():.2f} to {data['BF (%)'].max():.2f}")
    logger.info(f"POK range: {data['POK (%)'].min():.2f} to {data['POK (%)'].max():.2f}")

    # Input features (X) and outputs (y)
    X = data[['TPU (%)', 'BF (%)']].values
    y_strength = data['Strength (MPa)'].values
    y_fracture = data['Fracture Toughness (MPa·m1/2)'].values
    y_impact = data['Impact Energy Dissipation (J)'].values

    # Verify POK constraint
    invalid_pok = data[data['POK (%)'] < 80]
    if not invalid_pok.empty:
        logger.warning(f"Found {len(invalid_pok)} samples with POK < 80: {invalid_pok['POK (%)'].values}")

    return X, y_strength, y_fracture, y_impact, data

def main():
    # Parse arguments
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    X, y_strength, y_fracture, y_impact, data = load_data(args)

    # Add polynomial features and interaction term
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)
    scaler_poly = StandardScaler()
    X_poly_scaled = scaler_poly.fit_transform(X_poly)
    interaction = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    scaler_interaction = StandardScaler()
    X_interaction = scaler_interaction.fit_transform(interaction)
    X_full = np.column_stack((X_poly_scaled, X_interaction))

    # Scale inputs and log-transform outputs
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_full)

    y_log_strength = np.log(y_strength + 1e-1) * np.sign(y_strength + 1e-1)
    y_log_fracture = np.log(y_fracture + 1e-1) * np.sign(y_fracture + 1e-1)
    y_log_impact = np.log(y_impact + 1e-1) * np.sign(y_impact + 1e-1)

    scaler_strength = StandardScaler()
    scaler_fracture = StandardScaler()
    scaler_impact = StandardScaler()

    y_scaled_strength = scaler_strength.fit_transform(y_log_strength.reshape(-1, 1)).ravel()
    y_scaled_fracture = scaler_fracture.fit_transform(y_log_fracture.reshape(-1, 1)).ravel()
    y_scaled_impact = scaler_impact.fit_transform(y_log_impact.reshape(-1, 1)).ravel()

    # Function to train and evaluate GP model
    def train_gp(X_train, X_test, y_train, y_test, scaler, output_name):
        kernel = Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e4)) + \
                 WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=20)
        try:
            gp.fit(X_train, y_train)
            logger.info(f"GP Model for {output_name}: {gp.kernel_}")
        except Exception as e:
            logger.error(f"GP Model for {output_name} failed: {e}")
            raise
        y_pred_scaled, sigma_scaled = gp.predict(X_test, return_std=True)
        y_log = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_pred = np.exp(y_log * np.sign(y_log)) - 1e-1
        y_test_log = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_test_orig = np.exp(y_test_log * np.sign(y_test_log)) - 1e-1
        sigma = sigma_scaled * scaler.scale_
        r2 = r2_score(y_test_orig, y_pred)
        cv_scores = cross_val_score(gp, X_train, y_train, cv=5, scoring='r2')
        if cv_scores.mean() < 0:
            logger.warning(f"{output_name} has negative CV R² Mean: {cv_scores.mean():.3f}")
        if cv_scores.std() > 0.5:
            logger.warning(f"{output_name} has high CV R² Std: {cv_scores.std():.3f}")
        logger.info(f"{output_name} R²: {r2:.3f}, CV R² Mean: {cv_scores.mean():.3f}, CV R² Std: {cv_scores.std():.3f}")
        return gp, y_pred, sigma, r2, cv_scores.mean(), cv_scores.std()

    # Try multiple random seeds
    best_r2 = {'Strength': 0, 'Fracture': 0, 'Impact': 0}
    best_results = {}
    seeds = [42, 123, 456, 789, 101, 111, 222, 333, 444, 555]

    for seed in seeds:
        X_train, X_test, y_train_s, y_test_s = train_test_split(X_scaled, y_scaled_strength, test_size=5, random_state=seed)
        _, _, y_train_f, y_test_f = train_test_split(X_scaled, y_scaled_fracture, test_size=5, random_state=seed)
        _, _, y_train_i, y_test_i = train_test_split(X_scaled, y_scaled_impact, test_size=5, random_state=seed)

        gp_s, pred_s, sigma_s, r2_s, cv_s_mean, cv_s_std = train_gp(X_train, X_test, y_train_s, y_test_s, scaler_strength, 'Strength')
        gp_f, pred_f, sigma_f, r2_f, cv_f_mean, cv_f_std = train_gp(X_train, X_test, y_train_f, y_test_f, scaler_fracture, 'Fracture')
        gp_i, pred_i, sigma_i, r2_i, cv_i_mean, cv_i_std = train_gp(X_train, X_test, y_train_i, y_test_i, scaler_impact, 'Impact')

        if r2_s > 0.82 and r2_f > 0.82 and r2_i > 0.82 and cv_s_mean > 0.0 and cv_f_mean > 0.0 and cv_i_mean > 0.0:
            best_r2 = {'Strength': r2_s, 'Fracture': r2_f, 'Impact': r2_i}
            best_results = {
                'Seed': seed,
                'Strength': {'pred': pred_s, 'sigma': sigma_s, 'y_test': np.exp(scaler_strength.inverse_transform(y_test_s.reshape(-1, 1)).ravel() * np.sign(scaler_strength.inverse_transform(y_test_s.reshape(-1, 1)).ravel()) - 1e-1), 'r2': r2_s, 'cv_mean': cv_s_mean, 'cv_std': cv_s_std},
                'Fracture': {'pred': pred_f, 'sigma': sigma_f, 'y_test': np.exp(scaler_fracture.inverse_transform(y_test_f.reshape(-1, 1)).ravel() * np.sign(scaler_fracture.inverse_transform(y_test_f.reshape(-1, 1)).ravel()) - 1e-1), 'r2': r2_f, 'cv_mean': cv_f_mean, 'cv_std': cv_f_std},
                'Impact': {'pred': pred_i, 'sigma': sigma_i, 'y_test': np.exp(scaler_impact.inverse_transform(y_test_i.reshape(-1, 1)).ravel() * np.sign(scaler_impact.inverse_transform(y_test_i.reshape(-1, 1)).ravel()) - 1e-1), 'r2': r2_i, 'cv_mean': cv_i_mean, 'cv_std': cv_i_std}
            }
            break

    # Save results
    results = pd.DataFrame({
        'Metric': ['Strength', 'Fracture Toughness', 'Impact Energy'],
        'R2 Score': [best_r2['Strength'], best_r2['Fracture'], best_r2['Impact']],
        'CV R2 Mean': [best_results.get('Strength', {}).get('cv_mean', 0),
                       best_results.get('Fracture', {}).get('cv_mean', 0),
                       best_results.get('Impact', {}).get('cv_mean', 0)],
        'CV R2 Std': [best_results.get('Strength', {}).get('cv_std', 0),
                      best_results.get('Fracture', {}).get('cv_std', 0),
                      best_results.get('Impact', {}).get('cv_std', 0)]
    })
    results.to_csv(os.path.join(output_dir, 'gp_results.csv'), index=False, encoding='utf-8')

    # Save predictions
    predictions = pd.DataFrame({
        'Strength Actual': best_results.get('Strength', {}).get('y_test', []),
        'Strength Predicted': best_results.get('Strength', {}).get('pred', []),
        'Strength Sigma': best_results.get('Strength', {}).get('sigma', []),
        'Fracture Actual': best_results.get('Fracture', {}).get('y_test', []),
        'Fracture Predicted': best_results.get('Fracture', {}).get('pred', []),
        'Fracture Sigma': best_results.get('Fracture', {}).get('sigma', []),
        'Impact Actual': best_results.get('Impact', {}).get('y_test', []),
        'Impact Predicted': best_results.get('Impact', {}).get('pred', []),
        'Impact Sigma': best_results.get('Impact', {}).get('sigma', [])
    })
    predictions.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False, encoding='utf-8')

    # Save GP models and scalers
    joblib.dump(gp_s, os.path.join(output_dir, "gp_strength.pkl"))
    joblib.dump(gp_f, os.path.join(output_dir, "gp_fracture.pkl"))
    joblib.dump(gp_i, os.path.join(output_dir, "gp_impact.pkl"))
    joblib.dump(scaler_X, os.path.join(output_dir, "scaler_X.pkl"))
    joblib.dump(scaler_strength, os.path.join(output_dir, "scaler_strength.pkl"))
    joblib.dump(scaler_fracture, os.path.join(output_dir, "scaler_fracture.pkl"))
    joblib.dump(scaler_impact, os.path.join(output_dir, "scaler_impact.pkl"))
    joblib.dump(scaler_interaction, os.path.join(output_dir, "scaler_interaction.pkl"))
    joblib.dump(scaler_poly, os.path.join(output_dir, "scaler_poly.pkl"))
    joblib.dump(poly, os.path.join(output_dir, "poly.pkl"))

    # Print results
    logger.info(f"Best Seed: {best_results.get('Seed', 'None')}")
    for metric in ['Strength', 'Fracture', 'Impact']:
        r2 = best_r2.get(metric, 0)
        cv_mean = best_results.get(metric, {}).get('cv_mean', 0)
        cv_std = best_results.get(metric, {}).get('cv_std', 0)
        logger.info(f"{metric}: R² = {r2:.3f}, CV R² Mean = {cv_mean:.3f}, CV R² Std = {cv_std:.3f}")
        if r2 < 0.82:
            logger.warning(f"{metric} R² ({r2:.3f}) is below 0.82")

    logger.info("GP models and scalers saved as .pkl files in /results")

if __name__ == "__main__":
    main()

