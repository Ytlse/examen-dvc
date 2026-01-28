import sys
from pathlib import Path
src_path = str(Path(__file__).resolve().parents[1]) 
sys.path.append(src_path)
import os
import pickle
from xgboost import XGBRegressor
import pandas as pd
import logging
from data.check_structure import check_existing_folder

verbose = True

def fit(input_folderpath, X_train_filename, y_train_filename, best_params_path, output_model_folderpath):
    """
    Train a final XGBoost regression model using best parameters and save it.
    This function loads training data and pre-optimized hyperparameters, trains a final
    XGBoost regression model, and saves it to disk.
    Parameters
    ----------
    input_folderpath : str
        Path to the folder containing training data files.
    X_train_filename : str
        Filename of the training features CSV file.
    y_train_filename : str
        Filename of the training target CSV file.
    best_params_path : str
        Path to the pickle file containing the best hyperparameters.
    output_model_folderpath : str
        Path to the folder where the trained model will be saved.
    Returns
    -------
    None
        The trained model is saved to disk at output_model_folderpath/xgboost_final_model.pkl
    Raises
    ------
    FileNotFoundError
        If input data files or best_params_path do not exist.
    Notes
    -----
    - The model uses 'reg:squarederror' as the objective function and random_state=2026.
    - Output folder is created if it does not exist.
    - Requires 'verbose' to be defined in the calling scope for status messages.
    """

    # Import training data
    if verbose: print("Importing training data for grid search...")
    X_train_filepath = os.path.join(input_folderpath, X_train_filename)
    X_train = pd.read_csv(X_train_filepath, sep=",")
    y_train_filepath = os.path.join(input_folderpath, y_train_filename)
    y_train = pd.read_csv(y_train_filepath, sep=",")

    # Charger les meilleurs paramètres depuis le fichier .pkl
    if verbose: print("Loading best parameters from", best_params_path)
    with open(best_params_path, 'rb') as file:
        best_params = pickle.load(file)
    
    # Créer et entraîner le modèle final avec ces paramètres
    if verbose: print("Training final model with best parameters...")
    final_model = XGBRegressor(**best_params, objective='reg:squarederror', random_state=2026)
    final_model.fit(X_train, y_train) 

    # Sauvegarder le modèle final
    if verbose: print("Saving final model to", output_model_folderpath)
    if check_existing_folder(output_model_folderpath):
        os.makedirs(output_model_folderpath)

    final_model_path = os.path.join(output_model_folderpath, 'xgboost_final_model.pkl')
    with open(final_model_path, 'wb') as file:
        pickle.dump(final_model, file)
        if verbose: print("Saved final model to", final_model_path)


def main(input_folderpath="./data/processed_data/",
        X_train_filename = "X_train_scaled.csv",
        y_train_filename = "y_train.csv",
        best_params_path="./models/xgboost_best_params.pkl",
        output_model_folderpath="./models/"
        ):
    """ Run XGBoost final model training and save it to ./models/xgboost_final_model.pkl
    """
    fit(input_folderpath, X_train_filename, y_train_filename, best_params_path, output_model_folderpath)
    logger = logging.getLogger(__name__)
    logger.info('making scaled data set')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()