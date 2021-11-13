import pandas as pd
import os
import re
import numpy as np
import json

dir = r"D:\PEAV\AssaultVerdictsParameterExtraction\final_verdicts_dir"
SENTENCE_LEN = "length sentence"

def count_sentences(file):
    return file.count('.')

def count_words(file):
    return len(file.split())
    # return len(re.split("[\.!\?, /\n#]",file))
def count_sentences_lentgh(file):
    sentences = file.split('.')
    sentences_length = []
    for s in sentences:
        s_length = len(s.split())
        sentences_length.append(s_length)
    return sentences_length
    # print(len(sentences))
    # print("average sentence len = ", np.average(sentences_length))
    # print("variance sentence len = ", np.std(sentences_length))
    # print(max(sentences_length), " and min = ", min(sentences_length))
    # return np.min(sentences_length), np.max(sentences_length), np.average(sentences_length), np.var(sentences_length)

def iterate_files(file_list):
    sentences_num = 0
    word_num = 0
    file_num_counter = 0
    all_sentence_len = []
    for file_name in file_list:
        if file_name.endswith(".txt"):
            file_num_counter += 1
            f = open(dir+"\\"+file_name, encoding="utf8")
            file = f.read()
            all_sentence_len.extend(count_sentences_lentgh(file))
            sentences_num += count_sentences(file)
            word_num += count_words(file)

    print("amount of sentences = ", sentences_num)
    print("amount of words = ", word_num)
    print("amount of files = ", file_num_counter)

    print(len(all_sentence_len))
    print("average sentence len = ", np.average(all_sentence_len))
    print("variance sentence len = ", np.std(all_sentence_len))
    print(max(all_sentence_len), " and min = ", min(all_sentence_len))

def all_db_checks():
    with open('sentence_list.txt') as json_file:
        gzar_list = json.load(json_file)
    iterate_files(gzar_list)

def only_annotated():
    db = pd.read_csv(r"D:\PEAV\AssaultVerdictsParameterExtraction\db_csv_files\DB of 30.5.csv")
    sentences = db.sentence.to_list()
    sentences_length = []
    for s in sentences:
        s_length = len(s.split())
        sentences_length.append(s_length)
    print(len(sentences))
    print("average sentence len = ", np.average(sentences_length))
    print("variance sentence len = ", np.std(sentences_length))
    print(max(sentences_length), " and min = ", min(sentences_length))

def fleiss_kappa(M):
    """Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    # print("observed agreement = ",P)
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items
    # Pbar = P
    return round((Pbar - PbarE) / (1 - PbarE), 4)
    # return (Pbar - PbarE) / (1 - PbarE)


def cohen_kappa(ann1, ann2):
    """Computes Cohen kappa for pair-wise annotators.
    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list
    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count

    return round((A - E) / (1 - E), 4)

def run_fleiss_k (db):
    db = db.replace("Y", 1)
    db = db.replace("N", 0)
    db = db.replace("M", 2)
    mat = db.to_numpy()
    #
    print(fleiss_kappa(mat))

def run_inter_annotator_k(db):

    # each two:
    for i in range(5):
        for j in range(i+1,5):
            print("i,j = ",i,j)
            print(cohen_kappa(db.iloc[:,i].tolist(),db.iloc[:,j].tolist()))

if __name__ == "__main__":
    # all_db_checks()
    manual_tags_path = "D:\PEAV\AssaultVerdictsParameterExtraction\db_csv_files\mannual taggers by RK .csv"
    db = pd.read_csv(manual_tags_path,encoding='utf-8')
    # db = db.iloc[:, 5:8]
    run_inter_annotator_k(db)
