from bs4 import BeautifulSoup
import urllib#.request
import urllib3
# import hebpipe
import os
import re
import numpy as np
import pandas as pd
import json

VERDICTS_DIR = r"C:\Users\oryiz\PycharmProjects\PEAV\AssaultVerdictsParameterExtraction\after_extraction_verdicts\\"
VERDICTS_DIR = "final_verdicts_dir/"

DISTRICT = "district"
CASE_NUM = "case_num"
ACCUSED_NAME = "accused_name"
NUM_LINES = "num_lines"
DAY = "day"
MONTH = "month"
YEAR = "year"
COMPENSATION = "compensation"
CHARGES = "charges"
AGE = "age"
ARCHA = "archa"
JUDGE_NUM = "JUDGE_NUM"
FEMALE_J_NUM = "FEMALE_J_NUM"
MALE_J_NUM = "MALE_J_NUM"
FEMALE_J_PERCENT = "FEMALE_J_PERCENT"
ASSULTED_GENDER = "Assulted Gender"
VIRGINS = ""
IS_ANONYMOUS = "is anonymous"
CLOSED_DOOR = "is closed door"
IS_MINOR = "IS MINOR"

BAFOAL = 'בפועל'
ACTUAL_JAIL = 0
PROBATION = 1
COM_SERVICE = 2

YEAR_REG = r"([0-3]{0,1}[0-9]\/[0-2]{0,1}[0-9]\/[0-2]{0,1}[0-9]{1,3})|([0-3]{0,1}[0-9]\.[0-2]{0,1}[0-9]\.[0-2]{0,1}[0-9]{1,3})"
HEB_YEAR_EXP = "שנ(ים)*(ות)*|שנה|שנתיים"
HEB_MONTH_EXP = "ח(ו)*דש"
YEARS = "שנים"
MONTHS = "חודשים"

C345 = '345'
C346 = '346'
C347 = '347'
C348 = '348'
C349 = '349'
C350 = '350'
C351 = '351'

TIME_UNITS_ARR =  ["חודש", "שנה", "שנים", "שנות", "שעות","שנת","חדש"]
NUM_UNITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
         "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
         "אחד", "שתים", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע",
         "עשר", "אחת", "שנים", "שניים", "", "חמישה", "", "שישה"]

def vote_for_time(t, act_sent):
    """
    This function recieves a number and a sentence and votes about the likelyhood of it being the actual jail time.
    :param t: a number
    :param act_sent: the sentence which this number was taken from.
    :return: a score for this sentence
    """
    time_ind = act_sent.find(t)
    score = np.abs(act_sent.find(BAFOAL) - time_ind ) #Distance from bafoal
    ext = act_sent.find("יתר")
    page = act_sent.find("עמוד")
    if ext != -1:
        if time_ind < ext:
            score -= 10
    if np.abs (time_ind - page) < 10:
        score += 100

def find_time_act(act_sent):
    """

    :param act_sent:
    :return:
    """
    times = re.findall("[0-9]+", act_sent)
    times_and_dist = {}
    min_dist = 10000
    winner_time = '0'
    if len(times) > 2 and int(times[0]) == (int(times[1]) + int(times[2])):
        winner_time = times[1]

    else:
        for t in times:
            if act_sent.find("עמוד") != -1 and act_sent.find("עמוד") <= act_sent.find(t) <= (act_sent.find("עמוד") + 6):
                continue

            vote_for_time(t, act_sent)
            dist = np.abs(act_sent.find(BAFOAL) - act_sent.find(t))
            times_and_dist[t] = dist
            if dist < min_dist:
                min_dist = dist
                winner_time = t

            # reg = '\b(?:'+t+'\W+(?:\w+\W+){0,'+str(dist)+'}?'+BAFOAL+'|'+BAFOAL+'\W+(?:\w+\W+){0,'+str(dist)+'}?'+t+')\b'
    return times_and_dist, winner_time

def calc_punishment(sentence): #TODO - call this somewhere
    score_for_penalty = [0,0,0]

    if sentence.find("בפועל") != -1:
        score_for_penalty[ACTUAL_JAIL] += 3

    elif sentence.find("תנאי") != -1:
        score_for_penalty[PROBATION] += 3

    if sentence.find("שירות") != -1:
        if sentence.find("עבודות") != -1:
            score_for_penalty[COM_SERVICE] += 3

    return score_for_penalty

def calc_score(sentence):
    score_act = 0
    score_prob = 0

    list_of_bad_words = ['עו*תרה*(ים)*(ות)*','ה*תובעת*','ביקשה*','ה*תביעה','מבחן','צבאי','בי*טחון','קבע','דורשת*','בימים','בין','מתחם','יפחת','יעלה','נגזר','נדון','ה*צדדים']
    list_of_moderate_bad_words = ["\"","/","\\"] #":",
    list_of_good_words = ['גוזרת*(ים)*(ות)*','[נמ]טילה*(ים)*(ות)*',' ד[(נה)(ן)(נים)(נות)]','משיתה*','מחליטה*(ים)*(ות)*']
    list_of_moderate_good_words = ['לגזור','להטיל','יי*מצא מתאים']
    # if sentence.find() != -1 and sentence.find("\"") != -1 and (sentence.find("/") != -1 or sentence.find("\\")):  # maybe this is a quote?
    #     score_relevancy -=4
    #     print("shall not pass1 = ",sentence)

    if sentence.find("עונשין") != -1 and sentence.find("יעבור") != -1:
        score_act += 4

    for word in list_of_bad_words:
        if re.search(word,sentence):
            score_act -= 4
            score_prob -= 4

    for wr in list_of_moderate_bad_words:
        if sentence.find(wr) != -1:
            score_act -= 2
            if wr != "\"":
                score_prob += 2

    for w in list_of_good_words:
        if re.search(w,sentence):
            score_act +=4
            score_prob +=4

    for wr in list_of_moderate_good_words:
        if re.search(wr, sentence):
            score_act += 2
            score_prob += 2

    punishment = calc_punishment(sentence)
    score_act += punishment[0] + punishment[2]
    score_prob += punishment[1]

    return score_act, score_prob

def extracting_penalty(text, filename ):
    sentences = []
    len_sent = []
    penalty = "not found"
    main_sentence_act = "not found"
    main_sentence_prob = "not found"
    print("######################" + filename + "#######################")
    indices = [m.start() for m in re.finditer("מאסר", text)]

    for x in re.finditer("שירות", text):
        indices.append(x.start())

    for i in indices: #goes over all the indices of "maasar" in the text from last to first
        start = text.rfind(".", 0, i)
        end = text.find(".", i, len(text))
        sentence = text[start+1:end+1]
        for duration in TIME_UNITS_ARR:
            if sentence.find(duration) != -1:
                sentences.append(sentence)
                len_sent.append(end - start)
    all_times = 0
    prison_time = 0
    time_unit = "not found"
    if len(sentences) > 0:
        len_sent = len_sent[::-1]
        print("Sentences = ", sentences)
        max_score_act = -10
        max_score_prob = -10
        for i, sentence in enumerate(sentences[::-1]):
            scr_act, scr_prob = calc_score(sentence)
            scr_act = scr_act/len_sent[i]

            if scr_act > max_score_act:
                max_score_act = scr_act
                main_sentence_act = sentence
            if scr_prob > max_score_prob:
                max_score_prob = scr_prob
                main_sentence_prob = sentence

        # main_sentence_act = replace_value_with_key(main_sentence_act) #TODO notice This is turned off for sentence validation purposes
        all_times, prison_time = find_time_act (main_sentence_act)
        time_unit = find_time_units(main_sentence_act)

        if time_unit == YEAR:
            prison_time = float(prison_time)*12
            time_unit = YEAR

    return penalty, main_sentence_act, main_sentence_prob, all_times, prison_time,time_unit #כל מה שהפונקציה מחזירה שיהיה אח כך ב DB

def find_time_units(act_sent):
    if re.search(HEB_MONTH_EXP, act_sent):
        return MONTH
    elif re.search(HEB_YEAR_EXP, act_sent):
        return YEAR


def fromVerdictsToDB():
    """
    This function creates the feature db.
    :param df:
    :return:
    """
    batch = pd.DataFrame()
    directory = VERDICTS_DIR               #text files eddition:
    counter = 0

    with open('test_case_filenames.txt') as json_file:
        relevant_cases = json.load(json_file)
        for i, filename in enumerate(os.listdir(directory)):  # when iterating through all files in folder

            # for i in range(len(files)):                     # when iterating through all files in igud colum
            #     if type(files[i]) == str:
            #         filename = add_to_txt_db(files[i], urlToText(files[i]), "mechozi")

            if filename.endswith(".txt") and filename in relevant_cases:
                counter += 1
                file_name = os.path.join(directory, filename)
                text = open(file_name, "r", encoding="utf-8").read()

                print("^^^ File is ", file_name, " ^^^ - not psak"," counter = " ,counter)
                if filename.find("00001581-359.txt") != -1:
                    print("break point")

                main_penalty, sentence, all_sentences, all_times, time, time_unit = extracting_penalty(text, filename) #Here I call the penalty func

                batch.loc[i,"PENALTY_SENTENCE"] = main_penalty
                batch.loc[i,"VOTED TIME"] = time

                sentence_line = pd.DataFrame([[filename,"Gzar", main_penalty,     sentence,          all_sentences,   all_times,       time, time_unit]], #here I add values to DB
                                             columns =[CASE_NUM,     "TYPE","Main Punishment","PENALTY_SENTENCE", "ALL SENTENCES", "OPTIONAL TIMES", "VOTED TIME", "units"]) #Here adding a title
                db = pd.concat([db,sentence_line ])

            else:
                continue

            db.to_csv('verdict_penalty.csv', encoding= 'utf-8')

if __name__ == "__main__":

    # district_dict = {}
    with open('data.txt') as json_file:
        district_dict = json.load(json_file)
    #
    with open('county_list.txt') as json_file:
        county_dict = json.load(json_file)
    #
    with open('sentence_list.txt') as json_file:
        gzar_list = json.load(json_file)
    #

    with open('verdict_list.txt') as json_file:
        verdicts_list = json.load(json_file)

    fromVerdictsToDB()