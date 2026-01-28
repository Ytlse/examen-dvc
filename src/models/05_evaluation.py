import sys
from pathlib import Path
src_path = str(Path(__file__).resolve().parents[1]) 
sys.path.append(src_path)
import os
import pickle
import pandas as pd
import logging
from pathlib import Path
from data.check_structure import check_existing_file, check_existing_folder
from sklearn.metrics import r2_score, root_mean_squared_error
import json

verbose = True

def evaluate(X_test_filepath, y_test_filepath, best_model_path, pred_folderpath, metrics_folderpath):
    """
    Evaluate a trained machine learning model on test data and save results.
    This function loads a pre-trained model, evaluates its performance on test data,
    and saves predictions and metrics to specified directories.
        Parameters
        ----------
        input_folderpath : str
            Path to the folder containing test data files.
        X_test_filename : str
            Filename of the test features (X_test) CSV file.
        y_test_filename : str
            Filename of the test labels (y_test) CSV file.
        best_model_path : str
            Path to the pickled pre-trained model file.
        pred_folderpath : str
            Path to the folder where predictions will be saved.
        metrics_folderpath : str
            Path to the folder where evaluation metrics will be saved.
        Returns
        -------
        None
            The function prints evaluation results and saves predictions and metrics
            to the specified output directories.
        Notes
        -----
        - Test data is loaded from CSV files in the input folder.
        - The model is expected to be a pickled scikit-learn compatible model.
        - Evaluation metrics include R² score and Root Mean Squared Error (RMSE).
        - Predictions are saved as 'y_pred.csv' in pred_folderpath.
        - Metrics are saved as 'scores.json' in metrics_folderpath.
        """

    # Import testing data
    if verbose: print("Importing testing data for grid search...")
    X_test = pd.read_csv(X_test_filepath, sep=",")
    y_test = pd.read_csv(y_test_filepath, sep=",")

    # Load model
    if verbose: print("Loading best model from", best_model_path)
    with open(best_model_path, 'rb') as file:
        model = pickle.load(file)

    # Evaluate model
    if verbose: print("Evaluating model...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Score R² sur le test set : {r2:.4f}")
    print(f"RMSE sur le test set : {rmse:.4f}")
    
   # Sauvegarde des predictions
    if verbose: print("Saving predictions to", pred_folderpath)
    if check_existing_folder(os.path.dirname(pred_folderpath)):
        os.makedirs(os.path.dirname(pred_folderpath))

    predictions_filepath = os.path.join(pred_folderpath, 'y_pred.csv')
    if check_existing_file(predictions_filepath):
        pd.DataFrame(y_pred, columns=y_test.columns).to_csv(predictions_filepath, index=False)
        if verbose: print("Saved predictions to", predictions_filepath)

    # Sauvegarde des metrics
    if verbose: print("Saving metrics to", metrics_folderpath)
    if check_existing_folder(metrics_folderpath):
        os.makedirs(metrics_folderpath)

    final_metrics_path = os.path.join(metrics_folderpath, 'scores.json')
    with open(final_metrics_path, 'w') as file:
        metrics = {
            'r2_score': r2,
            'rmse': rmse
        }
        json.dump(metrics, file)
        if verbose: print("Saved metrics to", final_metrics_path)


def main(X_test_filepath = "./data/scaled_data/X_test_scaled.csv",
        y_test_filepath = "./data/processed_data/y_test.csv",
        best_model_path="./models/last/xgboost_final_model.pkl",
        pred_folderpath="./data/predictions/",
        metrics_folderpath="./metrics/"
        ):
    """ Evaluate model and save predictions and metrics
    """
    print("""-------------
          05 Starting final model evaluation...
          -------------""")
    evaluate(X_test_filepath,y_test_filepath, best_model_path, pred_folderpath,metrics_folderpath)
    logger = logging.getLogger(__name__)
    logger.info('making scaled data set')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()