import requests
import os
import logging
from check_structure import check_existing_file, check_existing_folder
import pandas as pd
from sklearn.model_selection import train_test_split

verbose = True

def import_raw_data(raw_data_relative_path, filename, bucket_folder_url):
    """
    Downloads a file from a specified bucket URL and saves it to a local directory.
    Parameters:
        raw_data_relative_path (str): The relative path to the local directory where the file will be saved.
        filename (str): The name of the file to download.
        bucket_folder_url (str): The base URL of the bucket or remote folder containing the file.
    Behavior:
        - Checks if the target directory exists; if not, creates it.
        - Constructs the full URL to the file and the local output path.
        - If the file does not already exist locally, downloads it from the bucket URL.
        - Saves the downloaded file content to the specified local path.
        - Prints status messages during the process.
        - Handles HTTP errors by printing an error message.
    Note:
        This function assumes the existence of helper functions `check_existing_folder` and `check_existing_file`.
    """

    '''import filenames from bucket_folder_url in raw_data_relative_path'''
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)
    # download all the files

    input_file = os.path.join(bucket_folder_url,filename)
    output_file = os.path.join(raw_data_relative_path, filename)
    if check_existing_file(output_file):
        object_url = input_file
        print(f'downloading {input_file} as {os.path.basename(output_file)}')
        response = requests.get(object_url)
        if response.status_code == 200:
            # Process the response content as needed
            content = response.text
            text_file = open(output_file, "wb")
            text_file.write(content.encode('utf-8'))
            text_file.close()
        else:
            print(f'Error accessing the object {input_file}:', response.status_code)

def process_data(input_folderpath, input_filename, output_folderpath):
    """
    Processes a dataset by importing it, splitting it into training and testing sets, and saving the resulting dataframes to the specified output folder.
    Args:
        input_folderpath (str): Path to the folder containing the input file.
        input_filename (str): Name of the input CSV file to be processed.
        output_folderpath (str): Path to the folder where the processed data will be saved.
    Workflow:
        1. Imports the dataset from the specified input file.
        2. Drops the 'date' column and splits the data into features and target ('silica_concentrate').
        3. Splits the data into training and testing sets (70% train, 30% test).
        4. Creates the output folder if it does not exist.
        5. Saves the resulting dataframes (X_train, X_test, y_train, y_test) as CSV files in the output folder.
    Notes:
        - Assumes the existence of a 'silica_concentrate' column in the input data.
        - Uses helper functions `check_existing_folder` and `check_existing_file` to manage file and folder creation.
        - Uses a global or external `verbose` variable to control logging output.
        - Requires pandas, os, and train_test_split from sklearn to be imported.
    """
    
    # Import datasets
    if verbose: print('Importing dataset...')
    input_filepath = os.path.join(input_folderpath, input_filename)
    df = pd.read_csv(input_filepath, sep=",")
    if verbose: 
        print('Dataset imported with shape:', df.shape)
        print(df.head())

        
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate','date'], axis=1)
    if verbose: print('Splitting data into training and testing sets...')
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)

    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

    # Save dataframes to output folder
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if verbose: print(f'Saving {filename} to {output_filepath}...')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

def main(input_filename = "raw.csv",
        bucket_folder_url= " https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/",
        input_folderpath="./data/raw_data/",
        output_folderpath="./data/processed_data/"          
        ):
    """ Upload data from AWS s3 in ./data/raw
    """
    print("""-------------
          01 Starting data processing...
          -------------""")
    import_raw_data(input_folderpath, input_filename, bucket_folder_url)
    process_data(input_folderpath,input_filename, output_folderpath,)
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()