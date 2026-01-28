dvc init
dvc stage add -n processed -d src/data/01_processed_data.py -d data/raw_data  -o data/processed_data/ python src/data/01_processed_data.py
dvc stage add -n scaled -d src/data/02_scaled_data.py -d data/processed_data  -o data/scaled_data/ python src/data/02_scaled_data.py
dvc stage add -n gridsearch -d src/models/03_model_gridsearch.py -d data/processed_data -d data/scaled_data/ -o models/gridsearch/ python src/models/03_model_gridsearch.py
dvc stage add -n fit -d src/models/04_model_fit.py -d data/processed_data -d data/scaled_data/  -o models/last/ python src/models/04_model_fit.py
dvc stage add -n evaluate -d src/models/05_evaluation.py -d data/processed_data -d data/scaled_data  -d models/last -o data/predictions -M metrics/scores.json python src/models/05_evaluation.py

05_evaluation.py
def main(X_test_filepath = "./data/scaled_data/X_test_scaled.csv",
        y_test_filepath = "./data/processed_data/y_test.csv",
        best_model_path="./models/last/xgboost_final_model.pkl",
        pred_folderpath="./data/predictions/",
        metrics_folderpath="./metrics/"
        ):