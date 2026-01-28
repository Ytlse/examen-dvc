import os
import logging
from check_structure import check_existing_file, check_existing_folder
import pandas as pd
from sklearn.preprocessing import StandardScaler


verbose = True

def scale_data(input_folderpath, input_train_filename, input_test_filename, output_folderpath):
    """
    Scales training and test datasets using standard normalization and saves the scaled data to the specified output folder.
    Parameters:
        input_folderpath (str): Path to the folder containing the input CSV files.
        input_train_filename (str): Filename of the training dataset CSV file.
        input_test_filename (str): Filename of the test dataset CSV file.
        output_folderpath (str): Path to the folder where the scaled datasets will be saved.
    Process:
        - Loads the training and test datasets from the specified input folder.
        - Fits a StandardScaler on the training data and applies the transformation to both training and test datasets.
        - Saves the scaled datasets as CSV files ('X_train_scaled.csv' and 'X_test_scaled.csv') in the output folder.
        - Creates the output folder if it does not exist.
    Notes:
        - Assumes the existence of helper functions `check_existing_folder` and `check_existing_file`.
        - Uses a global or external `verbose` variable for optional logging.
        - Requires pandas, numpy, os, and sklearn's StandardScaler.
    """
    
    # Import datasets
    if verbose: print('Importing X_train dataset...')
    input_filepath = os.path.join(input_folderpath, input_train_filename)
    df_train = pd.read_csv(input_filepath, sep=",")
    if verbose: 
        print('Dataset imported with shape:', df_train.shape)
        print(df_train.head())

    if verbose: print('Normalizing data...')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train)
    df_X_train_scaled =  pd.DataFrame(X_train_scaled, columns=df_train.columns)

    if verbose: print('Importing X_test dataset...')
    input_filepath = os.path.join(input_folderpath, input_test_filename)
    df_test = pd.read_csv(input_filepath, sep=",")
    if verbose: 
        print('Dataset imported with shape:', df_test.shape)
        print(df_test.head())

    X_test_scaled = scaler.transform(df_test)
    df_X_test_scaled = pd.DataFrame(X_test_scaled, columns=df_test.columns)

    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

    # Save dataframes to output folder
    for file, filename in zip([df_X_train_scaled, df_X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if verbose: print(f'Saving {filename} to {output_filepath}...')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

def main(input_folderpath="./data/processed_data/",
        input_train_filename = "X_train.csv",
        input_test_filename = "X_test.csv",
        output_folderpath="./data/scaled_data/"
        ):
    """ Scale data in ./data/processed_data
    """
    print("""-------------
          02 Starting data scaling...
          -------------""")
    scale_data(input_folderpath, input_train_filename, input_test_filename, output_folderpath)
    logger = logging.getLogger(__name__)
    logger.info('making scaled data set')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()