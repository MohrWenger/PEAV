from AssaultVerdictsParameterExtraction import penalty_extraction
import pandas as pd
import numpy as np
import json

FILE_NAME = "filename"
SENTENCE = "sentence"
SENTENCE_LEN = "length sentence"
RB_PRED_COL = "RB prediction"
# TAG_COL = "is correct sentence"
TAG_COL = "does sentence include punishment"
# TAG_COL = "original tag"
def calculate_sentences_len(sentences):
    sentences_len = []
    for s in sentences:
        sentences_len.append(len(s.split()))
    return sentences_len

def adding_RB_pred_to_db (tagged_db):
    tagged_db[RB_PRED_COL] = np.zeros(len(tagged_db))  # FIll the RB row with zeros
    verdicts_included = np.unique(tagged_db[FILE_NAME])

    for ver in verdicts_included:
        # if ver == "s02004853-273.txt":
        #     print('bp')
        temp_db = tagged_db.loc[tagged_db[FILE_NAME] == ver]
        sentences_len = temp_db[SENTENCE_LEN].tolist()
        sentences = temp_db[SENTENCE].tolist()
        if 0 in sentences_len:
            sentences_len = calculate_sentences_len(sentences)
        params = penalty_extraction.extract_penalty_params(sentences, sentences_len )
        chosen_sentence = params[1]
        # print("pred = ", chosen_sentence)
        if "not found" not in chosen_sentence:
            row = tagged_db.index[(tagged_db[FILE_NAME] == ver) & (tagged_db[SENTENCE] == chosen_sentence)] # The row with the chosen sentence
            # tagged_db[tagged_db.index == row[0]][RB_PRED_COL] = 1
            print("ver = ", ver)

            tagged_db.at[row[0],RB_PRED_COL] = 1

        # print("sanity check: ",chosen_sentence == tagged_db.iloc[row[0]][SENTENCE])
        # print()
    return tagged_db

def classic_f1(tagged_db):
    goal = tagged_db[TAG_COL].to_numpy()
    pred = tagged_db[RB_PRED_COL].to_numpy()
    true_positive = sum(pred[goal == 1])  # all the goal = 1 and then pred = 1
    false_negative = len(pred[goal == 1]) - true_positive  # all the pred = 0 where goal = 1
    false_positive = sum(pred[goal == 0])  # all the pred = 1 where goal was actually 0
    f1 = true_positive / (true_positive + 0.5 * (false_positive + false_negative))
    print ("num of 1 is ", sum(goal)," for ", len(np.unique(tagged_db[FILE_NAME]))," cases")
    print("f1 = ", f1)

def check_per_verdict(tagged_db):
    """
    Since the RB outputs one sentence per verdict we check if this sentence is one of the sentences tagged correct
    (and not how many of the correct sentences are found).
    :param tagged_db: The DB with our manual tagging and the RB tag.
    :return:
    """
    pred = tagged_db[RB_PRED_COL].to_numpy()
    goal = tagged_db[TAG_COL].to_numpy()
    ver_num = len(np.unique(tagged_db[FILE_NAME]))
    tp = np.sum(goal[pred == 1])
    fp = np.sum(pred) - tp
    fn = np.sum(goal[pred == 0])
    recall = tp/(tp+fn)
    percision = tp/(tp+fp)
    print("TP:",tp)
    print("FN:",fn)
    print("Sensitivity:",recall)
    print("percision:",percision)
    print("F1 = ", 2*recall*percision/(recall+percision))


def compare_selected_cases_per_DB(path):
    with open('test_case_filenames.txt') as json_file:
        fvpt_cases = json.load(json_file)

    db = pd.read_csv(path,encoding= 'utf-8')
    sst_cases = np.unique(db[FILE_NAME])
    print("sst = ",len(sst_cases), " and len fvpt = ",len(fvpt_cases))
    conj = []
    for case in fvpt_cases:
        if case in sst_cases:
            conj.append(case)
        else:
            print(case)

    print(len(conj))

def subset_of_tagged_correct(db):
    rb_decision = db.loc[db[RB_PRED_COL] == 1]
    rb_decision.to_csv("db_csv_files/RB_prediction_db.csv", encoding = 'utf-8')

if __name__ == "__main__":
    # path = r"D:/PEAV/AssaultVerdictsParameterExtraction/full_svm_prediction.csv"
    path = r"D:/PEAV/AssaultVerdictsParameterExtraction/db_csv_files/feature_DB 28.07.csv"
    # path = r"D:/PEAV/AssaultVerdictsParameterExtraction/db_csv_files/just tagged sentences.csv"
    tagged_db = pd.read_csv(path,header= 0, na_values='')
    print(len(tagged_db))
    # tagged_db = tagged_db.loc[tagged_db["does number appear"] == True]
    # tagged_db = tagged_db.iloc[:,0:6]
    tagged_db = adding_RB_pred_to_db(tagged_db)
    subset_of_tagged_correct(tagged_db)
    tagged_db.to_csv('db_csv_files/DB with RB.csv', encoding= 'utf-8')
    classic_f1(tagged_db)
    check_per_verdict(tagged_db)
    compare_selected_cases_per_DB(path)

