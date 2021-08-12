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

if __name__ == "__main__":
    all_db_checks()
