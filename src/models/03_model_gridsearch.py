import sys
from pathlib import Path
src_path = str(Path(__file__).resolve().parents[1]) 
sys.path.append(src_path)
import os
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import logging
from data.check_structure import check_existing_file, check_existing_folder

verbose = True

def xgboost_gridsearch(input_folderpath, X_train_filename, y_train_filename, output_folderpath):
    """
    Perform grid search for XGBoost hyperparameter tuning and save the best parameters.
    This function loads training data, performs a grid search with cross-validation
    to find the optimal hyperparameters for an XGBoost regressor, and saves the
    best parameters to a pickle file.
    Parameters
    ----------
    input_folderpath : str
        Path to the folder containing the training data files.
    X_train_filename : str
        Filename of the feature matrix (X_train) CSV file.
    y_train_filename : str
        Filename of the target variable (y_train) CSV file.
    output_folderpath : str
        Path to the folder where the best parameters will be saved.
    Returns
    -------
    None
        The function saves the best parameters to a pickle file in the output folder
        and prints status messages to console.
    Notes
    -----
    - The grid search uses 5-fold cross-validation with R² as the evaluation metric.
    - The parameter grid includes: n_estimators, max_depth, learning_rate, subsample,
      and colsample_bytree.
    - The output file is named 'xgboost.pkl' and contains the best_params_ dictionary
      from GridSearchCV.
    - Requires global variable 'verbose' to be defined for print statements.
    Raises
    ------
    FileExistsError
        Printed message if output file already exists (file is not overwritten).
    """

    # Import training data
    if verbose: print("Importing training data for grid search...")
    X_train_filepath = os.path.join(input_folderpath, X_train_filename)
    X_train = pd.read_csv(X_train_filepath, sep=",")
    y_train_filepath = os.path.join(input_folderpath, y_train_filename)
    y_train = pd.read_csv(y_train_filepath, sep=",")

    # Define the model
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],  # Nombre d'arbres
        'max_depth': [3, 5, 7],          # Profondeur maximale des arbres
        'learning_rate': [0.01, 0.1, 0.2],  # Taux d'apprentissage
        'subsample': [0.8, 1.0],         # Fraction des échantillons utilisés pour chaque arbre
        'colsample_bytree': [0.8, 1.0]   # Fraction des features utilisées pour chaque arbre
    }

    # Set up the GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='r2',  # Metric for evaluation
        verbose=1,
        n_jobs=-1  # Utilise tous les cœurs du CPU
    )

    # Fit the grid search to the data
    if verbose: print("Starting grid search for XGBoost hyperparameters...")
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres et le score
    if verbose: print("Meilleurs paramètres:", grid_search.best_params_)
    if verbose: print("Meilleur score R²:", grid_search.best_score_)

    # 5. Sauvegarder les meilleurs paramètres dans un fichier .pkl
    
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

    # Save best parameters
    output_filepath = os.path.join(output_folderpath, 'xgboost_best_params.pkl')
    if check_existing_file(output_filepath):
        with open(output_filepath, 'wb') as file:
            pickle.dump(grid_search.best_params_, file)
        print(f"Meilleurs paramètres sauvegardés dans {output_filepath}")
    else:
        print(f"Erreur : Le fichier {output_filepath} existe déjà.")


def main(input_folderpath="./data/processed_data/",
        input_Xtrain_filename = "X_train_scaled.csv",
        input_ytrain_filename = "y_train.csv",
        output_folderpath="./models/"
        ):
    """ Run XGBoost gridsearch and save best parameters in ./models/xgboost.pkl
    """
    xgboost_gridsearch(input_folderpath, input_Xtrain_filename, input_ytrain_filename, output_folderpath)
    logger = logging.getLogger(__name__)
    logger.info('making scaled data set')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()