import nltk
import sklearn_crfsuite
import sklearn_crfsuite.metrics as metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


TAG_COL = "does sentence include punishment"
TAG_PROB = 'probation'
SENTENCE = "sentence"
FILE_NAME = "filename"


def arrange_for_crf(df,labels):  # NOTICE - this assumes no shuffle in the data
    sent_in_file_dict = []
    all_files_lists = []
    prev_file_name = -1
    prev_row = {}
    sent_labels_lst = []
    all_labels_lst = []
    is_BOS = False
    counter = 0
    for i, row in df.iterrows():
        if counter == 0:
            prev_row = row
            prev_file_name = row[FILE_NAME]
            # sent_labels_lst.append(str(labels[counter]))

        else:
            prev_row_dict = dict(**prev_row)  # each time we want to update the previous row.
            if is_BOS:  # This happens if the previous line was the last in the file.
                prev_row_dict['BOS'] = True
                is_BOS = False

            if prev_file_name != row[FILE_NAME] or counter == df.shape[1]:  # if the next line file name is not the same then this row is the EOS.
                prev_file_name = row[FILE_NAME]  # update that we are in a new file
                prev_row_dict['EOS'] = True
                sent_in_file_dict.append(prev_row_dict)  # adding the dict of the last sentence
                sent_labels_lst.append(str(labels[i]))
                is_BOS = True
                all_files_lists.append(sent_in_file_dict)  # add the list of all dicts to the big list
                all_labels_lst.append(sent_labels_lst)  # add the labels to the big list

                sent_in_file_dict = []  # initialize both lists
                sent_labels_lst = []

            else:  # for rows that are either first or last
                sent_in_file_dict.append(prev_row_dict)
                sent_labels_lst.append(str(labels[i]))

            prev_row = row #update prev row
        counter += 1
    return all_files_lists, all_labels_lst


def remove_strings (db):
    # db = db.replace(db[SENTENCE].tolist(), db.index.tolist())  # This is the line that replaces sentences
    # db.insert(0,"new col", db.index.tolist() )                                                              # with indx.
    db = db.set_index(pd.Index(np.arange(len(db[SENTENCE]))))                                                              # with indx.
    le = LabelEncoder()
    file_name_dict = le.fit(db[FILE_NAME])
    db[FILE_NAME] = file_name_dict.fit_transform(db[FILE_NAME])

    return db, file_name_dict.classes_

def add_line(new_db, line, tag_name, prediction, db):
        sentence = db.sentence[line]
        # print(sentence)
        # line = db.loc[db[SENTENCE] == sentence]
        line = db.iloc[line]
        if prediction == 1:
            add = pd.DataFrame(
                [[line[FILE_NAME], sentence, line[tag_name], prediction]],
                columns=["file", "sentence", "original tag", "our tag"])
            new_db = pd.concat([new_db, add])
        return new_db


def using_crfsuite(x_db, tag, tag_name, df, test = True):
    x_db, file_name_dict = remove_strings(x_db)

    np.random.seed(42)
    temp_db = x_db.loc[:, x_db.columns != SENTENCE]
    x_train, x_test, y_train, y_test = train_test_split(temp_db, tag, shuffle=False, test_size=0.2, random_state=42)
    x_train_dict, y_train_lst = arrange_for_crf(x_train, y_train)

    crf = sklearn_crfsuite.CRF(
        algorithm='pa',
        # c1=0.1,
        # c2=0.1,
        # max_iterations=100,
        # all_possible_transitions=True,
        # all_possible_states = True
    )
    print(len(x_train_dict))
    print(len(y_train_lst))
    crf.fit(x_train_dict, y_train_lst)

    if test:
        x, y = arrange_for_crf(x_test, y_test)
    else:
        x = x_train_dict
        y = y_train_lst

    # create predictions of which are the correct sentences
    # predicted_results = crf.predict(x)
    # goal_labels = y.to_numpy()
    # original_indices = x.index.to_numpy()
    labels = list(crf.classes_)
    print("labels = ", labels)
    y_pred = crf.predict(x)
    for i, lab in enumerate(y):
        lab = np.array(lab)
        if '1' in lab:
            correct = np.argwhere(lab == '1')[0]
            print(y_pred[i][int(correct)])
        else:
            print("goal had no punisment for this one")
    print_sent_ones(y, y_pred, x, file_name_dict, df)
    print(metrics.flat_f1_score(y, y_pred,
                          average='weighted', labels=labels))
    # check_prediction(predicted_results, goal_labels, tag_name,original_indices, df)

def print_sent_ones(prediction, goals, x_df, file_name_list, main_df):
    file_names = {}
    for i, pred in enumerate(prediction):
        for j, l in enumerate(pred):
            if l == '1':
                name = x_df[i][j][FILE_NAME]
                if name in file_names.keys():
                    file_names[name].append(j)
                else:
                    file_names[name] = []
                    file_names[name].append(j)

    print(file_names)

    for f_num in file_names.keys():
        name = file_name_list[int(f_num)]
        file_df = main_df.loc[main_df[FILE_NAME] == name]

        print("\nfile: ", name)
        for v in file_names[f_num]:
            print(file_df.iloc[v][SENTENCE])
    print(np.unique(file_names))

def check_prediction(predicted_results, goal_labels,tag_name ,X, df):

    count_same = 0
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    ones = pd.DataFrame()
    for i in range(len(predicted_results)):
        if predicted_results[i] == goal_labels[i]:
            count_same += 1
            if predicted_results[i] == 1:
                true_positive += 1
                ones = add_line(ones, X[i], tag_name, predicted_results[i], df)

                # print(X[i])
                # print(df.sentence[X[i]])
                # print(df[tag_name][X[i]], "and goals = ", goal_labels[i])

            elif predicted_results[i] == 0:
                true_negative += 1

        elif goal_labels[i] == 1:
                false_negative +=1
        else:
            false_positive += 1
            ones = add_line(ones, X[i], tag_name, predicted_results[i], df)

            # print(df.sentence[X[i]])
    print("sample size: ", len(goal_labels))
    print("how many ones expected:", sum(goal_labels))
    print("how many ones predicted: ", sum(predicted_results))
    print("Precision: ", (true_positive*weights)/(true_positive*weights + false_positive))
    print("Recall: ", true_positive/sum(goal_labels))
    print("F1 score = ", calc_F1(true_positive*weights, true_negative, false_negative*weights, false_positive))

def calc_F1 (true_positive, true_negative, false_negative, false_positive):
    return true_positive/(true_positive+0.5*(false_positive+false_negative))

if __name__ == "__main__":
    path = r"D:\PEAV\AssaultVerdictsParameterExtraction\DB of 16.5 - Sheet1.csv"
    # path = "/Users/tomkalir/Projects/PEAV/AssaultVerdictsParameterExtraction/feature_DB - feature_DB (1).csv"
    # path = r"C:\Users\נועה וונגר\PycharmProjects\PEAV\AssaultVerdictsParameterExtraction\feature_DB - feature_DB (1).csv"
    db_initial = pd.read_csv(path, header=0, na_values='')
    db_filtered = db_initial
    x_db = db_filtered.loc[:, db_filtered.columns != TAG_COL]
    x_db = x_db.loc[:, x_db.columns != TAG_COL]
    tag = db_filtered[TAG_COL]
    using_crfsuite(x_db, tag, TAG_COL, db_filtered)

    # from csv import reader
    # import re
    # output = open("/cs/usr/tomka/PycharmProjects/yapproj/src/yap/input.txt", "w")
    # with open('DB of 16.5 - Sheet1.csv', 'r') as read_obj:
    #     csv_reader = reader(read_obj)
    #     list_of_rows = list(csv_reader)
    #     for row in list_of_rows:
    #         sentence = row[2]
    #         sep_sen = re.findall(r"[\w']+|[.,!?;-]", sentence)
    #         text = "\n".join(sep_sen)
    #         output.write(text + "\n\n")
    # output.close()
    #
