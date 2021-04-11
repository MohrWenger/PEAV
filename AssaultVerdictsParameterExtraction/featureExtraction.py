import re
import os
import pandas as pd
from AssaultVerdictsParameterExtraction.penalty_extraction import extracting_penalty_sentences as ext_sent
import json
from tqdm import tqdm

COL_NAMES = ["case_num", "JAIL", "PROBATION", "COM SERV", "REQUEST_1", "REQUEST_2","REQUEST_3", "PROCEQUTION", "EXAM", "MILITARY",
             "SAFE_SERVICE", "KEVA", "DAYS", "BETWEEN", "MITHAM", "REDUCED", "UPPER_LIMIT", "DERIVED", ""]

# Order of copying: list of bad words, list of bad signs list of good words list of moderate words
HEB_WORDS_TO_EXTRACT = ['עו*תרה*(ים)*(ות)*','ה*תובעת*','ביקשה*','ה*תביעה','מבחן','צבאי','בי*טחון','קבע','דורשת*','בימים',
                        'בין','מתחם','יפחת','יעלה','נגזר','נדון','ה*צדדים',"\"","/",r"\\",":",'גוזרת*(ים)*(ות)*',
                        '[נמ]טילה*(ים)*(ות)*',' ד[(נה)(ן)(נים)(נות)]','משיתה*','מחליטה*(ים)*(ות)*','לגזור','להטיל',
                        'יי*מצא מתאים']


def extract_important_words(sentence, words):
    list_of_indices = []
    for word in words:
        indices = [index.start() for index in re.finditer(word, sentence)]
        list_of_indices.append(indices)
    return list_of_indices


# def extracting_penalty(text):
#     sentences = []
#     len_sent = []
#     indices = [m.start() for m in re.finditer("מאסר", text)]
#     for x in re.finditer("שירות", text):
#         indices.append(x.start())
#     for i in indices:  # goes over all the indices of "maasar" in the text from last to first
#         start = text.rfind(".", 0, i)
#         end = text.find(".", i, len(text))
#         sentence = text[start+1:end+1]
#         sentences.append(sentence)
#         len_sent.append(end - start)
#     return sentences, len_sent


def calc_score(sentence):
    score_act = 0
    score_prob = 0

    list_of_bad_words = ['עו*תרה*(ים)*(ות)*','ה*תובעת*','ביקשה*','ה*תביעה','מבחן','צבאי','בי*טחון','קבע','דורשת*','בימים','בין','מתחם','יפחת','יעלה','נגזר','נדון','ה*צדדים']
    list_of_moderate_bad_words = ["\"","/","\\"] #":",
    list_of_good_words = ['גוזרת*(ים)*(ות)*','[נמ]טילה*(ים)*(ות)*',' ד[(נה)(ן)(נים)(נות)]','משיתה*','מחליטה*(ים)*(ות)*']
    list_of_moderate_good_words = ['לגזור','להטיל','יי*מצא מתאים']

    if sentence.find("עונשין") != -1 and sentence.find("יעבור") != -1:
        score_act += 4
        # print("shall not pass2 = ", sentence)

    for word in list_of_bad_words:

        if re.search(word,sentence):
            score_act -= 4
            score_prob -= 4
            # print("shall not pass3 = ", sentence)
            # print("bacuse of ",word)

    for wr in list_of_moderate_bad_words:
        # print("wr = ", wr)
        # if re.search(wr, sentence):
        if sentence.find(wr) != -1:
            score_act -= 2
            if wr != "\"":
                score_prob += 2
            # print("shall yes pass1 = ", sentence)
            # print("thanks to ", wr)

    for w in list_of_good_words:
        if re.search(w,sentence):
            score_act +=4
            score_prob +=4
            # print("shall yes pass1 = ", sentence)
            # print("thanks to ",w)

    for wr in list_of_moderate_good_words:
        if re.search(wr, sentence):
            score_act += 2
            score_prob += 2
            # print("shall yes pass1 = ", sentence)
            # print("thanks to ", wr)

    punishment = calc_punishment(sentence)
    score_act += punishment[0] + punishment[2]
    score_prob += punishment[1]

    return score_act, score_prob


def calc_punishment(sentence):
    score_for_penalty = [0,0,0]
    if sentence.find("בפועל") != -1:
        score_for_penalty[ACTUAL_JAIL] += 3
        # return 3
        penalty = "מאסר בפועל"
        found = True
        main_sentence = sentence
        # print(sentence)

    elif sentence.find("תנאי") != -1:
        score_for_penalty[PROBATION] += 3
        # return 0
        penalty = "מאסר על תנאי"
        found = True
        print(sentence)
        main_sentence = sentence

    if sentence.find("שירות") != -1:
        # return 3
        # if sentence.find("מבחן") == -1:
        #     score_for_penalty[COM_SERVICE] -= 1

        if sentence.find("עבודות") != -1:
            score_for_penalty[COM_SERVICE] += 3

    return score_for_penalty

def operating_func(filename):
    text = open(path + filename, "r", encoding="utf-8").read()
    sentences, len_sentences = ext_sent(text)
    for i in range(len(sentences)):
        important_words_list = extract_important_words(sentences[i], HEB_WORDS_TO_EXTRACT)
        sentence_line = pd.DataFrame(
            [[filename, len_sentences[i]] + important_words_list],
            # here I add values to DB
            columns=["filename", "length"] + HEB_WORDS_TO_EXTRACT)
    return sentence_line

def extract(directory, running_opt):
    featureDB = pd.DataFrame()

    if running_opt == 0:
        with open('test_case_filenames.txt') as json_file:
            relevant_cases = json.load(json_file)  # Cases of the validation file
            for i, filename in tqdm(enumerate(os.listdir(directory))):  # when iterating through all files in folder
                if filename.endswith(".txt") and filename in relevant_cases:
                    sentence_line = operating_func(filename)
                    featureDB = pd.concat([featureDB, sentence_line])

    elif running_opt == 2:
        counter = 0
        for filename in os.listdir(directory):
            if counter < 150:
                sentence_line = operating_func(filename)
                featureDB = pd.concat([featureDB, sentence_line])
                counter += 1

    featureDB.to_csv("feature_DB.csv", encoding="utf-8")
    return featureDB


path = "C:\\Users\\oryiz\\PycharmProjects\\PEAV\\AssaultVerdictsParameterExtraction\\final_verdicts_dir\\"
extract(path,2)





