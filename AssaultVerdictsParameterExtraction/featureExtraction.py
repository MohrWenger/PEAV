import re
import os
import pandas as pd

COL_NAMES = ["case_num", "JAIL", "PROBATION", "COM SERV", "REQUEST_1", "REQUEST_2","REQUEST_3", "PROCEQUTION", "EXAM", "MILITARY",
             "SAFE_SERVICE", "KEVA", "DAYS", "BETWEEN", "MITHAM", "REDUCED", "UPPER_LIMIT", "DERIVED", ""]

# Order of copying: list of bad words, list of bad signs list of good words list of moderate words
HEB_WORDS_TO_EXTRACT = ['עו*תרה*(ים)*(ות)*','ה*תובעת*','ביקשה*','ה*תביעה','מבחן','צבאי','בי*טחון','קבע','דורשת*','בימים',
                        'בין','מתחם','יפחת','יעלה','נגזר','נדון','ה*צדדים',"\"","/","\\",":",'גוזרת*(ים)*(ות)*',
                        '[נמ]טילה*(ים)*(ות)*',' ד[(נה)(ן)(נים)(נות)]','משיתה*','מחליטה*(ים)*(ות)*','לגזור','להטיל',
                        'יי*מצא מתאים']


def extracting_penalty(text, filename):
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
        # print(sentence)
        for duration in TIME_UNITS_ARR:
            if sentence.find(duration) != -1:
                sentences.append(sentence)
                len_sent.append(end - start)
    all_times = 0
    prison_time = 0
    time_unit = "not found"
    if len(sentences) > 0:
        # min_len = min(len_sent)
        len_sent = len_sent[::-1]
        print("Sentences = ", sentences)
        max_score_act = -10
        max_score_prob = -10
        for i, sentence in enumerate(sentences[::-1]):
            scr_act, scr_prob = calc_score(sentence)
            scr_act = scr_act/len_sent[i]
            # if len(sentence) == min_len:
            #     scr += 1
            if scr_act > max_score_act:
                max_score_act = scr_act
                main_sentence_act = sentence
            if scr_prob > max_score_prob:
                max_score_prob = scr_prob
                main_sentence_prob = sentence

        # main_sentence_act = replace_value_with_key(main_sentence_act) #TODO notice This is turned off for sentence validation purposes
        all_times, prison_time = find_time_act(main_sentence_act)
        time_unit = find_time_units(main_sentence_act)

        if time_unit == MONTH:
            prison_time = float(prison_time)/12
            time_unit = YEAR
            # print("score is ",scr)
        # print("max scr = ", max_score, "for sentence ",main_sentence)

        #     if not found:
        #         not_good += 1
        #     print("HERE")
        #     print(sentences)
        # print(all)
        # print(not_good)

    return all, not_good, penalty, main_sentence_act, main_sentence_prob, all_times, prison_time,time_unit #כל מה שהפונקציה מחזירה שיהיה אח כך ב DB


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


def extract(directory):
    featureDB = pd.DataFrame()
    for filename in os.listdir(directory):
        text = open(filename, "r", encoding="utf-8").read()
        featureDB = featureDB.append()

        COL_NAMES_TEMP = [""]


def extractImportantWords(sentence, words):
    list_of_indices = []
    for word in words:
        list_of_indices.append([m.span()[0] for m in re.finditer(word, sentence)])
    return list_of_indices


