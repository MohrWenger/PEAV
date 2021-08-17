import joblib
import pandas as pd
import numpy as np
from AssaultVerdictsParameterExtraction import crf_model
from AssaultVerdictsParameterExtraction import model_outline as svm_model

# TAG_COL = "does sentence include punishment"
TAG_PROB = 'probation'
SENTENCE = "sentence"
FILE_NAME = "filename"

def run_crf():
    model_path = 'D:/PEAV/AssaultVerdictsParameterExtraction/crf_trained_model.pkl'
    # file_path = r"D:\PEAV\AssaultVerdictsParameterExtraction\db_csv_files\maasar only less features 27.06.csv"
    file_path = r"feature_DB.csv"

    model = joblib.load(model_path)

    db_initial = pd.read_csv(file_path, header=0, na_values='')
    db_initial = db_initial.set_index(np.arange(len(db_initial)))
    db, file_name_dict = svm_model.remove_strings(db_initial)

    np.random.seed(42)
    all_files = np.unique(db[FILE_NAME])
    np.random.seed(0)
    np.random.shuffle(all_files)
    file_num = len(all_files)
    cross_chunks = [all_files[x:x + int(file_num / 9)] for x in range(0, file_num, int(file_num / 9))]
    print("chunk num = ", len(cross_chunks))
    recalls = []
    precisions = []
    f1_score = []
    x, y = crf_model.new_arrangment(db, np.unique(db[FILE_NAME]), tagged= False)

    y_preds = model.predict(x)
    crf_model.print_sent_ones(y_preds, x, file_name_dict, db_initial)

def run_SVM():
    model_path = 'D:/PEAV/AssaultVerdictsParameterExtraction/SVM_trained_model.pkl'
    file_path = r"D:\PEAV\AssaultVerdictsParameterExtraction\db_csv_files\feature_DB 28.07.csv"
    file_path = r"feature_DB.csv"

    model = joblib.load(model_path)

    db_initial = pd.read_csv(file_path, header=0, na_values='')
    db_initial = db_initial.set_index(np.arange(len(db_initial)))
    db, file_name_dict = crf_model.remove_strings(db_initial)

    np.random.seed(42)
    temp_db = db.loc[:, db.columns != SENTENCE]
    # x_train, x_test, y_train, y_test = train_test_split(temp_db, tag, shuffle=False, test_size=0.2, random_state=42)
    # x_train_dict, y_train_lst = arrange_for_crf(x_train,y_train)
    all_files = np.unique(db[FILE_NAME])
    np.random.seed(0)
    np.random.shuffle(all_files)
    file_num = len(all_files)
    cross_chunks = [all_files[x:x + int(file_num / 9)] for x in range(0, file_num, int(file_num / 9))]
    print("chunk num = ", len(cross_chunks))
    recalls = []
    precisions = []
    f1_score = []
    x = db.loc[:, db.columns != SENTENCE]
    original_indices = x.index.to_numpy()
    y_preds = model.predict(temp_db)
    y_proba = model.predict_proba(temp_db)
    ones = svm_model.smaller_sentence_pool(y_preds,TAG_PROB, original_indices,db_initial, y_proba, False)
    ones['length sentence'] = 0
    ones.to_csv("full_svm_prediction_trained.csv",encoding= "utf-8")
    svm_model.apply_argmax(ones, pd.DataFrame(),db, evaluate= False)

if __name__ == "__main__":
    run_SVM()