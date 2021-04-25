from bs4 import BeautifulSoup
import urllib#.request
import urllib3
# import hebpipe
import os
import re
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# VERDICTS_DIR = r"C:\Users\oryiz\PycharmProjects\PEAV\AssaultVerdictsParameterExtraction\after_extraction_verdicts\\"
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
HEB_YEAR_EXP = "שנ[ים]*[ות]*"  # remember that I took down שנה|שנתיים
HEB_MONTH_EXP = "חו*דש"
YEARS = "שנים"
MONTHS = "חודשים"

C345 = '345'
C346 = '346'
C347 = '347'
C348 = '348'
C349 = '349'
C350 = '350'
C351 = '351'

CLAUSES = [C345, C346, C347, C348, C349, C350, C351]

TIME_UNITS_ARR =  ["חודש", "שנה", "שנים", "שנות", "שעות","שנת","חדש"]
NUM_UNITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
         "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
         "אחד", "שתים", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע",
         "עשר", "אחת", "שנים", "שניים", "", "חמישה", "", "שישה"]

regex_numbers_heb = ["(אח[ד|ת]|לחודש|לשנה|לשנת)", "(שנתיי?ם|חודשיים|שתיי?ם|שניים)", "(שלושה?)", "(ארבעה?)", "(חמי?שה?)",
                     "(שישה|ששה?)", "(שי?בעה?)", "(שמונה?)", "(תשעה?)", "(עשרה?)"]
ten = "(\s?-?\s?עשרה?)?"
and_str = "(ו)?"
tens = "(ים)?"
unit_check = "(\s)"
half = "\s?וחצי\s?"


def numberExchange(sentence):
    """
    This function receives a sentence and replaces the word phrases for number with integers
    :param sentence:
    :return: A sentence after the replacement was done
    """
    i = 1
    might_be_unit = ""
    might_be_half = ""
    new_sentence = sentence
    for word in regex_numbers_heb:  # Go over the regexes and find number expressions
        pattern = and_str + word + tens + ten + unit_check  # Creating a regex with groups according to hebrew language rules
        heb_numbers = [m.span() for m in re.finditer(pattern, sentence)]  # A list of full matches and not groups
        # print(pattern)
        # print(heb_numbers)
        for tup in heb_numbers:
            value = 0
            match = re.match(pattern, sentence[tup[0]:tup[1]])  # This is meant to find the groups within a previuosly found pattern

            if match.group(3):  # Finding if there is an hebrew suffix for tens (if found multiply).
                if word == regex_numbers_heb[-1]:
                    value = 20
                else:
                    value = i * 10
                next = sentence[tup[1] + 1:].find("\s")
                for j in range(len(regex_numbers_heb)):  # Finding if there is a ones value and if so what is it.
                    pattern2 = and_str + regex_numbers_heb[j] + tens + ten
                    match2 = re.match(pattern2, sentence[tup[1] + 1:tup[1] + next + 1])
                    if match2 and match2.group(1):  # Checks whether there is indeed "and" before the digit.
                        pattern += "\s" + pattern2
                        value += j + 1
                        break

            if not match.group(3) and match.group(5):  # Then it is a number up until 10 (no prefix or suffix)
                if match.group(2) == "חודשיים":
                    might_be_unit = " חודשים"
                elif match.group(2) == "שנתיים":
                    might_be_unit = " שנים"
                elif not re.sub("לשנ[ה|ת]\s", "", match.group(0)):
                    might_be_unit = " שנה"
                value = i

            if match.group(4):  # Between 10 and 20
                value = i + 10
            # print(pattern + half)
            if re.search(pattern + half, new_sentence):
                value += 0.5
                might_be_half = half
            new_sentence = re.sub(pattern + might_be_half, str(value) + might_be_unit, new_sentence)
        i += 1
    return new_sentence

def vote_for_time(t, act_sent, t_units):
    """
    This function receives a number and a sentence and votes about the likelihood of it being the actual jail time.
    :param t: a number
    :param act_sent: the sentence which this number was taken from.
    :return: a score for this sentence
    """
    if act_sent.find((BAFOAL)) != -1:
        score = np.abs(act_sent.find(BAFOAL) - act_sent.find(t)) #starting from measuring the distance to the word of actual.
    else:  # TODO Maybe some flag but not sure?
        score = 0

    time_ind = act_sent.find(t)
    # Make sure it is not a page index.
    if act_sent.find("עמוד") != -1 and act_sent.find("עמוד") <= act_sent.find(t) <= (act_sent.find("עמוד") + 6):
        score += 100

    # Check if it is under the first section (א) TODO in choosing the sentence level

    #Make sure it is not a date:
    # dates = re.findall(YEAR_REG, act_sent)
    # if len(dates) > 0:
    #     for d in dates:
    #     print("date place = ", dates)

    #Distance from a time unit
    units = [(-1,-1)]
    if t_units == MONTH:
        units = re.findall(HEB_MONTH_EXP, act_sent)
    elif t_units == YEAR:
        units = re.findall(HEB_YEAR_EXP, act_sent)

    for u in units: #start from time_ind
        if type(u[1]) == str:
            ind = act_sent.find(u)
            if ind != -1 and np.abs(ind - time_ind) < 5:
                score -= 1000
                break

    #Words that it is less likely to appear after
    start_time = act_sent.find("יום")
    if start_time != -1 and time_ind < start_time:
        score += 10

    if len(act_sent) - time_ind < 4:
        score += 50

    return score


def find_time_act(act_sent):
    """
    This function receives a sentence and extracts the actual jail time given.
    :param act_sent:
    :return:
    """
    act_sent = numberExchange(act_sent)
    # act_sent = replace_value_with_key(act_sent)
    times = re.findall("([0-9]+(\.5)?)", act_sent) #finds all the numbers, assumes word numbers were converted.
    times = [x[0] for x in times] #takes only full matches
    times_and_score = {}
    min_score = 10000
    winner_time = '0'

    if len(times) > 2 and float(times[0]) == (float(times[1]) + float(times[2])): #A structure that returns: X1 time in total X2 actual and X3 probation
        winner_time = times[1]

    elif act_sent.find("יתר") != -1:
        winner_time = times[1]

    else:
        t_units = find_time_units(act_sent)
        for t in times:
            if (t_units == YEAR and float(t) > 20) or ( float(t) > 20 * 12) :
                continue

            else:
                score = vote_for_time(t, act_sent,t_units)
                times_and_score[t] = score
                if score < min_score:
                    min_score = score
                    winner_time = t

            # reg = '\b(?:'+t+'\W+(?:\w+\W+){0,'+str(dist)+'}?'+BAFOAL+'|'+BAFOAL+'\W+(?:\w+\W+){0,'+str(dist)+'}?'+t+')\b'
    return times_and_score, winner_time


def replace_value_with_key(sentence):
    """

    :param sentence:
    :return:
    """
    with open('nums_reg.txt') as json_file:
        num_dict = json.load(json_file)

    #pre check if there is number bigger then 10.

    for n in num_dict:
        sentence = re.sub(num_dict[n], n, sentence)

    return sentence


def calc_punishment(sentence):
    """
    This function chooses which of the punishments to score, according to which word appears
    :param sentence:
    :return:
    """
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
        if re.search(w, sentence):
            score_act += 4
            score_prob += 4

    for wr in list_of_moderate_good_words:
        if re.search(wr, sentence):
            score_act += 2
            score_prob += 2

    punishment = calc_punishment(sentence)
    score_act += punishment[0] + punishment[2]
    score_prob += punishment[1]

    return score_act, score_prob


def extracting_penalty_sentences(text, count_sentences=False):
    """
    This function is responsible for extracting the penalty sentences from the full verdict
    :param text: This function receives the full verdict's text
    :return: A list of all the potential relevant sentences and their lengths.
    """

    sentences = []
    len_sent = []
    sent_num = []
    indices = [m.start() for m in re.finditer("שירות|מאסר", text)]

    last_dot = 0
    sentence_count = 0
    for i in indices: #goes over all the indices of "maasar" in the text from last to first
        start = text.rfind(".", 0, i)
        end = text.find(".", i, len(text))
        num_of_sentences_from_last = re.findall("\.", text[last_dot:end])

        sentence = text[start+1:end+1]
        if sentence not in sentences:
            # print(sentence_count)
            # print(sentence)
            sentences.append(sentence)
            len_sent.append(end - start)
            sentence_count += len(num_of_sentences_from_last)
            sent_num.append(sentence_count)
            last_dot = end

    if count_sentences:
        return sentences, len_sent, sent_num
    return sentences, len_sent

def extract_penalty_params(sentences, len_sentences):
    relevant_sentences = []
    relevant_sent_lens = []

    for i, sentence in enumerate(sentences): #probsbly there is a better way to account for this but didn't want to change to much.
        for duration in TIME_UNITS_ARR:
            if sentence.find(duration) != -1:
                relevant_sentences.append(sentence)
                relevant_sent_lens.append(len_sentences[i])

    penalty = "not found"
    main_sentence_act = "not found"

    all_times = 0
    prison_time = 0
    time_unit = "not found"
    if len(relevant_sentences) > 0:
        relevant_sent_lens = relevant_sent_lens[::-1]
        max_score_act = -10
        max_score_prob = -10
        for i, sentence in enumerate(relevant_sentences[::-1]):
            scr_act, scr_prob = calc_score(sentence)
            scr_act = scr_act / relevant_sent_lens[i]

            if scr_act > max_score_act:
                max_score_act = scr_act
                main_sentence_act = sentence
            if scr_prob > max_score_prob:
                max_score_prob = scr_prob
                main_sentence_prob = sentence

        all_times, prison_time, time_unit = extract_time_from_sentence(main_sentence_act)
    main_sentence_prob = numberExchange(main_sentence_act)
    return penalty, main_sentence_act, main_sentence_prob, all_times, prison_time,time_unit #כל מה שהפונקציה מחזירה שיהיה אח כך ב DB

def extract_time_from_sentence(sentence):
        # main_sentence_act = replace_value_with_key(sentence) #TODO notice This is turned off for sentence validation purposes
        main_sentence_act = sentence
        all_times, prison_time = find_time_act (main_sentence_act)
        time_unit = find_time_units(main_sentence_act)

        if time_unit == YEAR:
            prison_time = float(prison_time)*12
            time_unit = YEAR
        else:
            time_unit = MONTH

        return all_times, prison_time, time_unit

def find_time_units(act_sent):
    if re.search(HEB_MONTH_EXP, act_sent):
        return MONTH
    elif re.search(HEB_YEAR_EXP, act_sent):
        return YEAR

def add_to_txt_db(url, text, court_type, name_only = False): # currently edited to return name and not write
    # name_file = url.strip("https://www.nevo.co.il/psika_html/"+court_type+"/")
    if text == "":
        print("breakpoint at ",url)
    name_file = url.split("/")[-1]
    name_file = name_file.replace(".htm",".txt")
    if not name_only:
        with open(VERDICTS_DIR + name_file, "w", encoding="utf-8") as newFile:
            newFile.write(text)
    return name_file

def urlToText(url):
    print("url = ",url)
    # webUrl = urllib.urlopen("file://"+url)
    webUrl = urllib.request.urlopen(url)
    html = webUrl.read()
    # import urllib
    # webUrl = urllib.urlopen("https://www.nevo.co.il/psika_html/mechozi/ME-19-07-69765-55.htm").read()
    # html = html_1.decode('utf-8')
    soup = BeautifulSoup(html, features="html.parser", from_encoding= 'utf-8-sig')

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    # print(soup.original_encoding)
    # print(text)
    return text


def coreFromVerdicts(db, filename, directory):
    file_name = os.path.join(directory, filename)
    text = open(file_name, "r", encoding="utf-8").read()

    # print("^^^ File is ", file_name, " ^^^ - not psak", " counter = ", counter)
    if filename.find("SH-12-03-28879-11.txt") != -1:
        print("break point")
    sentence_list, len_sent = extracting_penalty_sentences(text)
    main_penalty, sentence, all_sentences, all_times, time, time_unit = extract_penalty_params(sentence_list, len_sent)  # Here I call the penalty func

    # batch.loc[i,"PENALTY_SENTENCE"] = main_penalty
    # batch.loc[i,"VOTED TIME"] = time

    sentence_line = pd.DataFrame(
        [[filename, "Gzar", main_penalty, sentence, all_sentences, all_times, time, time_unit]],
        # here I add values to DB
        columns=[CASE_NUM, "TYPE", "Main Punishment", "PENALTY_SENTENCE", "ALL SENTENCES", "OPTIONAL TIMES",
                 "VOTED TIME", "units"])  # Here adding a title
    db = pd.concat([db, sentence_line])
    return db


def fromVerdictsToDB(running_opt):
    """
    This function creates the feature db.
    :param df:
    :return:
    """
    db = pd.DataFrame()
    directory = VERDICTS_DIR               #text files eddition:
    counter = 0

    if running_opt == 0:
        with open('test_case_filenames.txt') as json_file:
            relevant_cases = json.load(json_file)  # Cases of the validation file
            for i, filename in tqdm(enumerate(os.listdir(directory))):  # when iterating through all files in folder
                if filename.endswith(".txt") and filename in relevant_cases:
                    counter += 1
                    print(counter)
                    db = coreFromVerdicts(db, filename, directory)

    elif running_opt == 1:
        batch = pd.read_csv("Igud_Gzar2 - Sheet1.csv", error_bad_lines=False)
        files = batch['קובץ']  # Take all htm urls as a list
        for i in range(len(files)):                     # when iterating through all files that are Gzar Dins
            if type(files[i]) == str:
                filename = add_to_txt_db(files[i], "urlToText(files[i])", "mechozi", name_only= True)
                db = coreFromVerdicts(db, filename, directory)

    elif running_opt == 2:
        for i, filename in tqdm(enumerate(os.listdir(directory))):  # when iterating through all files in folder
            if filename.endswith(".txt"):# and counter < 150:
                counter += 1
                db = coreFromVerdicts(db, filename, directory)
            else:
                continue

    db.to_csv('verdict_penalty.csv', encoding= 'utf-8')

def from_sentence_list(case_names, sentence_list):
    """
    This function is used for main penalty time from a list of sentences.
    :param case_names: A list of the cases names
    :param sentence_list: A list of the sentences corresponding to the list of sentences
    :return: Nothing, Exports a CSV file with the sentences and extracted times.
    """
    s_db = pd.DataFrame()
    for i in range(len(sentence_list)):

        print('i = ', i)
        if case_names[i] == "ME-09-09-8805-668.txt":
            print("b point")
        if type(sentence_list[i]) == str:
            all_times, prison_time, time_unit = extract_time_from_sentence(sentence_list[i])
            sentence_line = pd.DataFrame(
                [[case_names[i], sentence_list[i],all_times, prison_time, time_unit]],
                # here I add values to DB
                columns=[CASE_NUM,"PENALTY_SENTENCE","OPTIONAL TIMES","VOTED TIME", "units"])  # Here adding a title
            s_db = pd.concat([s_db, sentence_line])
        else:
            sentence_line = pd.DataFrame(
                [[case_names[i], sentence_list[i], " -- ", " -- ", " -- "]],
                # here I add values to DB
                columns=[CASE_NUM, "PENALTY_SENTENCE", "OPTIONAL TIMES", "VOTED TIME", "units"])  # Here adding a title
            s_db = pd.concat([s_db, sentence_line])

    s_db.to_csv('pipline on test set.csv', encoding='utf-8')

def pipline_on_test_set():
    test_set = pd.read_csv("Test Set - PEAV - Sheet1.csv", error_bad_lines=False)
    from_sentence_list(test_set["Case_Name"], test_set["עונש בפועל (המשפט הרלוונטי מהטקסט)"])

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

    fromVerdictsToDB(2)

    # pipline_on_test_set()