dvc init
dvc stage add -n processed -d src/data/01_processed_data.py -d data/raw_data/raw.csv  -o data/processed_data/X_train.csv -o data/processed_data/X_test.csv -o data/processed_data/y_train.csv -o data/processed_data/y_test.csv python src/data/01_processed_data.py
dvc stage add -n scaled -d src/data/02_scaled_data.py -d data/processed_data/X_train.csv -d data/processed_data/X_test.csv -o data/scaled_data/X_train_scaled.csv -o data/scaled_data/X_test_scaled.csv python src/data/02_scaled_data.py
dvc stage add -n gridsearch -d src/models/03_model_gridsearch.py -d data/processed_data/y_train.csv -d data/scaled_data/X_train_scaled.csv -o models/gridsearch/xgboost_best_params.pkl python src/models/03_model_gridsearch.py
dvc stage add -n fit -d src/models/04_model_fit.py -d data/processed_data/y_train.csv -d data/scaled_data/X_train_scaled.csv -d models/gridsearch/xgboost_best_params.pkl  -o models/last/xgboost_final_model.pkl python src/models/04_model_fit.py
dvc stage add -n evaluate -d src/models/05_evaluation.py -d data/processed_data/y_test.csv -d data/scaled_data/X_test_scaled.csv  -d models/last/xgboost_final_model.pkl -o data/predictions/y_pred.csv -M metrics/scores.json python src/models/05_evaluation.py

05_evaluation.py
def main(X_test_filepath = "./data/scaled_data/X_test_scaled.csv",
        y_test_filepath = "./data/processed_data/y_test.csv",
        best_model_path="./models/last/xgboost_final_model.pkl",
        pred_folderpath="./data/predictions/",
        metrics_folderpath="./metrics/"
        ):