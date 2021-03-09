from bs4 import BeautifulSoup
import urllib#.request
import urllib3
# import hebpipe
import os
import re
import string
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
import conllu
import json
import glob
import datefinder

VERDICTS_DIR = r"C:\Users\oryiz\PycharmProjects\PEAV\AssaultVerdictsParameterExtraction\after_extraction_verdicts\\"
# VERDICTS_DIR = "verdicts/"

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


# TA = ת"\א
# DIST_DICT = {"תל אביב":[],
#              "באר שבע":[]}

no_districtCounter = 0
no_courtCounter = 0
no_caseNameCounter = 0
no_chargesCounter = 0
no_compensCounter = 0
no_accusedName = 0
no_ageCounter = 0

#### CONVENTIONS
# All the extraction functions RETURN the value at the end, which will then be submitted into the DB
# If an extraction function doesn't find the parameter, we return -1

################# YAP #####################
"""verdicts = "verdicts/"
output = open("yap_inputs/input.txt", "w")
for filename in os.listdir("verdicts/"):
    file = open(verdicts + filename,"r").read()
    output.write(filename)
    sep_file = re.findall(r"[\w']+|[.,!?;-]", file)
    text = "\n".join(sep_file)
    correctly_spaced = text.replace(".", ".\n")
    output.write(correctly_spaced)
    output.write("\n")
    output.write("\n")
output.close()

file = open("/Users/tomkalir/Projects/yap/src/yap/output.conll","r").read()
num_of_male = len(re.findall("gen=M", file))
num_of_female = len(re.findall("gen=F", file))
print(num_of_female)
print(num_of_male)
"""
################# Generic Functions For Extraction ######################

def allTheTextAfterAWord(text, word, until=-1):
    targetStartIndex = text.find(word)
    return text[targetStartIndex:until]


def findBetweenParentheses(text):
    targetStartIndex = text.find("(") + 1
    parenthesisAfterEndIndex = text.find(")", targetStartIndex)
    return text[targetStartIndex:parenthesisAfterEndIndex]


def extractWordAfterKeywords(text, words, until=" ", numOfWords=1):
    for word in words:
        keywordIndex = text.find(word)
        if keywordIndex != -1:
            break
    if keywordIndex == -1:
        return -1
    targetStartIndex = keywordIndex + len(word) + 1  # plus one because of space
    i = 0
    lastSpace = targetStartIndex
    while i < numOfWords:
        spaceAfterEndIndex = text.find(until, lastSpace) + 1
        lastSpace = spaceAfterEndIndex
        i += 1
    print("text = ",text[(targetStartIndex-10):spaceAfterEndIndex - 1])
    return text[targetStartIndex:spaceAfterEndIndex - 1]

def get_lines_after(text, word, amount,startAfter, limit="eof"):
    """
    This function returns the amount of lines after the line in which the word appeard
    :param text:
    :param word:
    :param amount:
    :return:
    """
    t_by_lines = text.splitlines()
    num_lines = len(t_by_lines)
    print("looking for ",word," in verdict: ",t_by_lines[0])
    if limit == "eof":
        limit = len(t_by_lines)
    start = 0
    for line in range(len(t_by_lines)):
        if re.search(word, t_by_lines[line]):
            start = line + startAfter
            relevant_lines = []
            if start < limit:
                for i in range(amount):
                    if start + i < limit:
                        relevant_lines.append(t_by_lines[start + i])
                # print("\n".join(relevant_lines))
                return "\n".join(relevant_lines)
            else:
                return ""
    return -1

##############################PARAMETERS#######################################

def extractLaw(name,text):
    reg_word = "חקיקה שאוזכרה"
    # reg_word = "פסק ה*דין|הכרעו*ת (- )*דין"
    relevant = get_lines_after(text, reg_word, 10, 1)
    # print("relevant = ",relevant )
    all_charges = np.zeros(len(RELEVANT_CHARGES))
    amount_appeared = []
    try:
        for i,chrg in enumerate(RELEVANT_CHARGES):
            if relevant != -1:
                if re.search(relevant, chrg) != -1:
                    all_charges[i] = text.count(chrg)
                    # all_charges.append([chrg, text.count(chrg)])
                    # amount_appeared.append(text.count(chrg))
            else:
                print("not find psak din in: ",name)
        print("section = ", all_charges)

    except:
        print("An exception occurred")

    # print("appeared: ", amount_appeared)
    return all_charges


def countWords(text, wordsList, numOfLines=0, wordAfter=""):
    all = []
    amount_appeared = []
    if wordAfter:
        text = get_lines_after(text, wordAfter, numOfLines, 0)
    for word in wordsList:
        if text.find(word) != -1:
            all.append([word, text.count(word)])
            amount_appeared.append(text.count(word))
    return all


def accusedName(text):
    accused_found = extractWordAfterKeywords(text, ["נ'"])
    return accused_found

def compensation(text):  # TODO: in one instance finds the salary instead of compensation
    # text = allTheTextAfterAWord(text, "סיכום") # TODO: not always a title of "summary"
    return extractWordAfterKeywords(text, ["סך של כ-", "סך של"]) # TODO: sometimes there's only "סך"

def courtArea(text):
    dist =  findBetweenParentheses(text)
    if any(i.isdigit() for i in dist):
        return -1
    else:
        return dist

def extract_dist_from_court(court):
    # print(district_dict)

    print("court = ", court)

    for dist in district_dict:
        # print("districts = ", district_dict[dist])
        if re.search(district_dict[dist], court):
            return dist

    return -1

def extract_county(dist, court):
    # print(county_dict)
    print("dist = ",dist)
    print("court = ",court)
    for c in county_dict:
        print(county_dict[c])
        if re.search(county_dict[c], dist) or re.search(county_dict[c], court):
            return c
    return -1

def replace_value_with_key(sentence):
    with open('nums_reg.txt') as json_file:
        num_dict = json.load(json_file)

    for n in num_dict:
        sentence = re.sub(num_dict[n], n, sentence)

    return sentence



def howManyLines(text):
    return len(text.split("."))


def judges(text):
    # lastIndex = text.rfind("שופט")
    # return countWords(text[lastIndex-100:lastIndex+5], ["שופטת", "שופט ", "שופט,", "נשיאה", "נשיא ", "נשיא,"])
    text = text.replace("נשיא", "שופט")
    instances = [i for i in range(len(text)) if text.startswith('שופט', i)]
    i = 1
    maxCount = 0
    while i < len(instances):
        count = 0
        while i < len(instances) and instances[i] - instances[i-1] < 50:
            count += 1
            i += 1
        if not count:
            i += 1
        if count > maxCount:
            maxCount = count
    return maxCount + 1


def sexOfJudges(text):
    text = text.replace("נשיא", "שופט")
    instances = [i for i in range(len(text)) if text.startswith('שופט', i)]
    i = 1
    maxCount = 0
    first = 0
    while i < len(instances):
        count = 0
        maybe = i
        while i < len(instances) and instances[i] - instances[i - 1] < 50:
            count += 1
            i += 1
        if not count:
            i += 1
        if count > maxCount:
            maxCount = count
            first = maybe

    judges = instances[first:first+maxCount+1]
    female = 0
    for judge in judges:
        if text[judge+4] != " ":
            female += 1
    return female


def ageOfVictim(text):
    found = extractWordAfterKeywords(text, [" כבת", " בת","טרם מלאו לה ","מתחת לגיל "])
    print("age = ", found)
    if found != -1:
        cleanFound = found.translate(str.maketrans('', '', string.punctuation))
        if cleanFound.isnumeric():
            return cleanFound
        found = extractWordAfterKeywords(text, [" ילידת"])
        if found != -1:
            cleanFound = found.translate(str.maketrans('', '', string.punctuation))
            if cleanFound.isnumeric():
                return cleanFound
    return -1

def interestingWords(text):
    print(text.find("אינה בתולה") != -1)
    print(ageOfVictim(text))
    # print(get_lines_after(text, "בוצעה*", 2, 0)) # interesting to find the year the accusation took place
    # print(extractWordAfterKeywords(text, [" כבת", " בת"]))

def isVictimMale(text):
    return True if text.count("מתלוננת") < text.count("מתלונן") else False


############ Main Functions ################
def extractParameters(text, db, case_name):
    #TODO : figure out how to limit the search area (ideas - number of lines, not in entioned laws, before discausion etc...)
    # think of a good structure to call each function of extraction and put the output in the correct column

    accused_name = accusedName(text)
    db = db.append({'accused_name': accused_name}, ignore_index=True)
    isAnonymous = False
    if accused_name == -1:
        global no_accusedName
        no_accusedName += 1
    else:
        accused_name == accused_name.split("\n")[0]

    if accused_name == "פלוני":
        isAnonymous = True
    # print("isAnonym type = ",type(isAnonymous))

    if (accused_name != "מדינת"):
        minor = is_minor(text)
        # print("minor type = ",type(minor))
        # charges = -1
        charges =  extractLaw(case_name,text)
        db = db.append({'charges':charges},ignore_index=True)

        if np.sum(charges) == 0:
            global no_chargesCounter
            no_chargesCounter += 1

        compens = compensation(text)
        db = db.append({'compensation':compens},ignore_index=True)
        if compens == -1:
            global no_compensCounter
            no_compensCounter += 1

        court = courtArea(text)
        district = -1
        level = -1
        county = -1
        db = db.append({"court": court},ignore_index=True)
        if court == -1:
            global  no_courtCounter
            no_courtCounter +=1
        else:
            district = extract_dist_from_court(court)
            # print("found dist - ", district)
            if district == -1:
                global no_districtCounter
                no_districtCounter += 1
                # print("not found dist - ",district)
                # print("len - ",len(court.split(" ")))
                if len(court.split(" "))> 1:
                    district = court.split(" ")[1:][0]
                else:
                    district = court.split(" ")[0]
            level = court.split(" ")[0]
            county = extract_county(district, court)

        age = ageOfVictim(text)
        db = db.append({AGE: age},ignore_index=True)
        if age == -1:
            global no_ageCounter
            no_ageCounter +=1

        lines_num = howManyLines(text)
        db = db.append({"lines_num": lines_num},ignore_index=True)

        date = extract_publish_dates(text)
        if date != "-1":
            # print()
            day, month, year = date
            day = int(day)
            month = int(month)
            year = int(year)
        else:
            day, month, year = [-1,-1,-1]
        db = db.append({YEAR: year},ignore_index=True)
        db = db.append({MONTH: month},ignore_index=True)
        db = db.append({DAY: day},ignore_index=True)
        db = db.append({"case_name": case_name},ignore_index=True)
        judges_amount = judges(text)
        female_J = -1
        sexPerc = -1
        if (judges_amount != -1):
            female_J = sexOfJudges(text)
            male_J = judges_amount - female_J
            sexPerc = round(female_J/judges_amount,3)

            db = db.append({"JUDGES NUM":judges_amount},ignore_index=True)
            db = db.append({"FEMALE_NUM":female_J},ignore_index=True)
            db = db.append({"FEMALE-percent":sexPerc},ignore_index=True)
        else:
            db = db.append({"JUDGES NUM": -1})
            db = db.append({"FEMALE_NUM": -1})
            db = db.append({"MALE_NUM": -1})
            db = db.append({"FEMALE-percent": -1})

        assultedGender = isVictimMale(text)
        if ( assultedGender != -1):
            db = db.append({"Assulted Gender":assultedGender},ignore_index=True)


        interestingWords(text)
        # ['case_num', 'year', 'court', 'charges', 'accused_name', 'lines_num'])
        # if np.sum(charges) != 0:
        case_ftr = pd.DataFrame([[case_name, day,month,  year,   court,  district, level, county , minor ,age,compens, accused_name, isAnonymous,
                                  assultedGender, judges_amount, female_J, male_J,sexPerc, lines_num,
                                  charges[0],charges[1],charges[2],charges[3],charges[4],charges[5],charges[6]]],

                                columns=['case_num',DAY, MONTH, YEAR, 'court',DISTRICT,'level','county', IS_MINOR,AGE ,'compensation','accused_name', IS_ANONYMOUS,
                                         ASSULTED_GENDER,JUDGE_NUM,FEMALE_J_NUM,MALE_J_NUM,FEMALE_J_PERCENT,NUM_LINES,
                                         C345,C346,C347,C348,C349,C350,C351])
        # db = pd..appended(case_ftr)
        return case_ftr

def createNewDB():
    # create a xls file with the right columns as the parameters
    # TODO: idea to use a dictionary as the structure to create this DB
    df = pd.DataFrame()
    # df = pd.DataFrame(columns=[CASE_NUM, DAY, MONTH, YEAR, DISTRICT, AGE ,CHARGES, COMPENSATION, ACCUSED_NAME, NUM_LINES])
    return df


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

def is_psak_din(filename, text):
    print("filename = ", filename)
    psak = False
    reg_word = "פסק ה*דין|הכרעו*ת (- )*דין"
    gzar_reg = "גזר ה*דין"
    file_name = re.compile(filename)
    if (re.search(reg_word,text)):
        # print("re search vals = ", re.search(gzar_reg, text))
        psak = True
    elif (re.search(gzar_reg,text)):
        # print("re search vals = ",re.search(gzar_reg,text))
        psak = False
    elif filter(file_name.match, verdicts_list):
        # print("re search vals not found",re.search(reg_word,text))
        psak = True
    elif filter(file_name.match, gzar_list):
        psak = False
    else:
        print("I didnt really know")
    return psak
        #         return False


def htmlToText():
    for filename in os.listdir("/Users/tomkalir/Downloads/igud1202/html/"):
        html = open("/Users/tomkalir/Downloads/igud1202/html/" + filename, "rb").read()
        soup = BeautifulSoup(html, features="html.parser", from_encoding='utf-8-sig')

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


        extracting_penalty(text, filename)

def calc_score(sentence):
    score_relevancy = 8
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



def calc_punishment(sentence): #TODO - call this somewhere
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
        # penalty = "עבודות שירות"
        # main_sentence = sentence
        # found = True
        # print(sentence)

def vote_for_time(t, act_sent):
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

def find_time_units(act_sent):
    if re.search(HEB_MONTH_EXP, act_sent):
        return MONTH
    elif re.search(HEB_YEAR_EXP, act_sent):
        return YEAR

def extracting_penalty(text, filename, all, not_good):
    sentences = []
    len_sent = []
    penalty = "not found"
    main_sentence_act = "not found"
    main_sentence_prob = "not found"
    # if text.find("גזר דין") != -1:
    all += 1
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
        all_times, prison_time = find_time_act (main_sentence_act)
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

        # if text.find("גזר דין") != -1:
        #     print("######################" + filename + "#######################")
        #     indices = [m.start() for m in re.finditer("מאסר", text)]
        #     for i in indices[-5:]:
        #         start = text.rfind(".", 0, i)
        #         end = text.find(".", i, len(text))
        #         sentence = text[start:end]
        #         if sentence.find("\"") != -1:
        #             continue
        #         if sentence.find("עונשין") != -1:
        #             continue
        #         found = False
        #         if sentence.find("בפועל") != -1:
        #             for duration in ["חודש", "שנה", "שנים", "שנות"]:
        #                 if sentence.find(duration) != -1:
        #                     found = True
        #             if found:
        #                 print(sentence)
        #                 break
        #         for duration in ["חודש", "שנה", "שנים"]:
        #             if sentence.find(duration) != -1:
        #                 found = True
        #         if found:
        #             print("HERE")
        #             print(text[start:end])
        #             break
# htmlToText()





# urlToText("https://www.nevo.co.il/psika_html/mechozi/ME-98-4124-HK.htm")
# urlToText("https://www.nevo.co.il/psika_html/mechozi/ME-17-12-378-33.htm")

#################### mohr current working on ###############################33
def convert_str_to_int_dict(str_arr):
        unique_vals = list(Counter(str_arr).keys())
        print(unique_vals)
        int_vals = np.arange(len(unique_vals))
        new_dict = {unique_vals[i]: int_vals[i] for i in range(len(unique_vals))}
        print(new_dict)
        return  new_dict

def is_eirur(text):
    name = text.splitlines()[0]
    # print("first line = ",name)
    if name.find("ע\"פ") != -1:
        return True
    else:
        return False

def is_minor(text):
    # word = "קטין|קטינה|קטינים|קטינות"
    reg_word = "חקיקה שאוזכרה"

    # reg_word = "פסק ה*דין|הכרעו*ת (- )*דין"
    relevent_text = get_lines_after(text, reg_word, 50, 2)

    #is female minor:
    minors_expressions = ["קטינות","קטינה","קטינים","קטין","דלתיים סגורות"]
    results = np.zeros(5)
    if relevent_text!= -1:
        for i, exp in enumerate(minors_expressions):
            if relevent_text.find(exp) != -1:
                results[i] = text.count(exp)
        print("minor exp = ",results)
        print("minor ret = ",True if np.sum(results) > 0 else False)
        return True if np.sum(results) > 0 else False
    else:
        # relevent_lines = text.splitlines()[:30]
        # relevent_text = "\n".join(relevent_lines)
        # print("30 first lines = ",)
        # for i, exp in enumerate(minors_expressions):
        #     if relevent_text.find(exp) != -1:
        #         results[i] = text.count(exp)
        # print("minor exp = ",results)
        # print("minor ret = ",True if np.sum(results) > 0 else False)
        # return True if np.sum(results) > 0 else False
        return "-1"


# def extractLaw(text):
#     reg_word = "חקיקה שאוזכרה"
#     # reg_word = "פסק ה*דין"
#     relevant = get_lines_after(text, reg_word, 10, 3)
#     all_charges = []
#     amount_appeard = []
#     line_appeadr =[] #TODO Deduce from the line all apeared in which is previuously mentioned
#     for chrg in RELEVANT_CHARGES:
#         if relevant != -1:
#             # print("chrg = ",type(chrg))
#             # if relevant.find(chrg) != -1:
#             if re.findall(text, chrg) != -1:
#                 all_charges.append([chrg, text.count(chrg)])
#                 amount_appeard.append(text.count(chrg))
#     print("section = ", all_charges)
#     # print("appeared: ", amount_appeard)
#     if len(all_charges) > 0:
#         return all_charges
#     else:
#         return -1

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

counter_noYearFound = 0

def extract_publish_dates(text):
    # t = get_lines_after(text, "נ'|נגד", 50,0) TODO - when the year is in paranthesis
    name = text.splitlines()[0].replace("(","")
    name = name.replace(")","")
    print("name = ",name)
    #print("ext = ", extractWordAfterKeywords(name, " בנבו, "))
    matches = re.findall(YEAR_REG,name)
    if len(matches) > 0:
        print("matches = ",matches[0][1])
        date = matches[0][1]
        print("date = ",date)
        day, month, year = date.split(".")
        return [(int)(day),(int)(month),(int)(year)]
    else:
        global counter_noYearFound
        counter_noYearFound += 1
        print("check here cause got -1")
        return "-1"

def get_urls_from_text_source(text_source):
    # text = open(text_source,"r")
    with open(text_source, 'r',encoding="utf8") as file:
        text = file.read()#.replace('\n', '')
    urls = []
    t_by_lines = text.splitlines()
    key_word = "<a title=\"הורדת HTML\" class=\"textLink htmLink"
    ref_start = "href=\""
    for line in t_by_lines:
        if (line.find(key_word) > 0):
            start = line.find(ref_start) + len(ref_start)
            end = line[start:].find("\" t")
            # print("indesx = ",start)
            urls.append("https://www.nevo.co.il"+line[start:start + end])
            # print("https://www.nevo.co.il"+line[start:start + end])
    print(urls)
    return urls

def htmlToText(dir, filename):

        html = open(dir + filename, "rb").read()
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

def get_all_URLS(source):
    content = urllib.request.urlopen(source)
    html = content.read()
    soup = BeautifulSoup(html, features="html.parser")
    div_list = []

    print("soup = ", soup.get_text())#"<div class=\"documentsLinks\">")

    for ul in soup.find_all('div'):
        div_list.extend(ul.find_all('div', {'class': 'searchItem'}))

    print(div_list)

    # print("soup = ", soup.find('div', { "role" : "article"}))#"<div class=\"documentsLinks\">")
"""
This function receives a path with many verdicts (presumably in word or html format), and uses the code to create a
database
"""
def fromVerdictsToDB():
    with open('test_case_filenames.txt') as json_file:
        relevant_cases = json.load(json_file)
    db = createNewDB()
    # case_names = []
    # compens = []
    # districts = []
    # charges = []
    # lines_number = []
    # all_accused = []
    batch = pd.read_csv("Igud_Gzar2 - Sheet1.csv", error_bad_lines = False)
    files = batch['קובץ']#Take all htm urls as a list
    directory = VERDICTS_DIR               #text files eddition:
    # years = []
    # years = [1998, 2006, 2009, 2006, 2006, 2004, 1999, 2000, 2005, 2000, 2015, 1994, 2001, 2016, 2005, 2001, 2001, 2001, 2003]
    counter = -1
    # print("years len = ", len(years))
    all = 0
    not_good = 0
    for i, filename in enumerate(os.listdir(directory)): #when iterateing through all files in folder
    # for i in range(len(files)): #when iterateing through all files in igud colum
    #     if type(files[i]) == str:
    #         filename = add_to_txt_db(files[i], urlToText(files[i]), "mechozi")
        if filename.endswith(".txt") and filename in relevant_cases:
            counter += 1
            file_name = os.path.join(directory, filename)
            text = open(file_name, "r", encoding="utf-8").read()
            # if (is_psak_din(filename,text)):
            #     pass
                # print("^^^ File is ", file_name, " ^^^")
                # # print("filename = ",filename,"counter = ",counter,"year = ",years[counter])
                # verd_line = extractParameters(text, db, filename)
                # if verd_line is not None:
                #     db = pd.concat([db,verd_line ])
            # else:
            print("^^^ File is ", file_name, " ^^^ - not psak"," counter = " ,counter)
                # if filename.find("00001295-97.txt") != -1:
                    # print("break point")
            all, not_good, main_penalty, sentence, all_sentences, all_times, time, time_unit = extracting_penalty(text, filename, all, not_good) #Here I call the penalty func
            batch.loc[i,"PENALTY_SENTENCE"] = main_penalty
            batch.loc[i,"VOTED TIME"] = time
            # batch.loc[i,"PENALTY_SENTENCE"] = main_penalty
            sentence_line = pd.DataFrame([[file_name,"Gzar", main_penalty,     sentence,          all_sentences,   all_times,       time, time_unit]], #here I add values to DB
                                         columns =[CASE_NUM,     "TYPE","Main Punishment","PENALTY_SENTENCE", "ALL SENTENCES", "OPTIONAL TIMES", "VOTED TIME", "units"]) #Here adding a title
            db = pd.concat([db,sentence_line ])



                # all_accused.append( accusedName(text))
            # #charges.append(extractLaw(text))
            # compens.append(compensation(text))
            # districts.append( courtArea(text))
            # lines_number.append(howManyLines(text))
            # case_names.append(filename)
        else:
            continue
    # db.to_csv('verdict_penalty.csv', encoding= 'utf-8')
    db.to_csv('verdict_penalty.csv', encoding= 'utf-8')
    # db = db.append(pd.concat([pd.DataFrame([all_accused[i], [case_names[i]]],
    #                                        columns=['accused']) for i in range(len(years))],ignore_index=True))
    # db = db.append(pd.concat([pd.DataFrame([case_names[i]],
    #                                        columns=['case_name']) for i in range(len(years))],ignore_index=True))
    # db = db.append(pd.concat([pd.DataFrame([districts[i]],
    #                                        columns=['district']) for i in range(len(years))],ignore_index=True))
    # db = db.append(pd.concat([pd.DataFrame([years[i]],
    #                                        columns=['year']) for i in range(len(years))],ignore_index=True))

        # for url in urls:                       #html edition
    #     text = urlToText(url)
    #     print(url)  # as kind of a title
    #     ExtractParameters(text, db)
        # add_to_txt_db(url, text,"mechozi")

    comp = [-1, -1, -1, 10000, 3500, -1, -1, 7000, -1, -1, -1, 3000, -1, 170000, 75000, -1, -1, -1, -1]
    places = ["שלום תל אביב-יפו", "מחוזי נצ'", "מחוזי מרכז",  "מחוזי י-ם", "שלום ירושלים", "פורסם בנבו, 10.10.1999",
                  "מחוזי תל אביב-יפו", "מחוזי חיפה", "מחוזי באר שבע", "מחוזי נצרת" , "מחוזי ב\"ש", "מחוזי חיפה",
                  "מחוזי נצרת", "מחוזי תל אביב-יפו", "מחוזי חיפה", "מחוזי תל אביב-יפו", "מחוזי תל אביב-יפו", "מחוזי באר שבע", "מחוזי תל אביב-יפו"]
        # tel_aviv = [1998, 2001, 2001, 2000, 2003]
    # plot_amount_of_param_in_param(db, "district","year")

    print("\n\n")

# source = "https://www.nevo.co.il/"
source = "https://www.nevo.co.il/PsikaSearchResults.aspx"
text_searc = "C:\\Users\\oryiz\\Desktop\\MohrsStuff\\URLs From Nevo\\search1.txt"
# get_all_URLS(source)
# urls = get_urls_from_text_source(text_searc)
RELEVANT_CHARGES = ['345', '346', '347', '348', '349', '350', '351']
searches_results = ["search15.txt","search16.txt","search17.txt","search18.txt","search19.txt","search20.txt"]
# searches_results = ["search11.txt","search12.txt","search13.txt","search14.txt","search15.txt","search16.txt","search17.txt","search18.txt","search19.txt","search20.txt"]

def from_search_to_local():
    dir = "SearchResults\\"
    for serach in searches_results:
        allURLS = get_urls_from_text_source(dir+serach)
        for url in allURLS:
            text = urlToText(url)
            if not is_eirur(text):
                print("url = ",url)
                if url.find("mechozi") > 0:
                     add_to_txt_db(url,text,"mechozi")
                elif url.find("shalom") > 0:
                    add_to_txt_db(url, text, "shalom")
                else:
                    print("didn't work for: ",url)
        print("finished with search: ", serach)


#------------------ Real plots ---------------------------------#
def plot_amount_per_param(batch, param,str_labels = False, should_revers = False, bar_plot = False, designated_lables = None):
    curr_batch = batch.loc[batch[param] != '-1']
    curr_batch = batch.loc[batch[param] != -1]
    all_param = np.unique(batch[param])
    amount_in_param = []
    x_lables = []
    for p in all_param:
        print("p = ",p)
        # if p == "פלוני" or p == "מדינת" :
        amount_in_param.append(len(curr_batch.loc[batch[param] == p]))
        if designated_lables == None:
            if should_revers:
                # p = p[::-1]
                x_lables.append(p[len(p):len(p)-5:-1])
            elif str_labels == False:
                x_lables.append(round(p,3))
            elif designated_lables == None:
                x_lables.append(p)
    if bar_plot == True:
        plt.bar(np.arange(len(amount_in_param)), amount_in_param,color = "palevioletred")
    else:
        plt.plot(np.arange(len(amount_in_param)), amount_in_param,color = "palevioletred")
    # plt.xlim(1985,2025)
    if designated_lables != None:
        x_lables = designated_lables
    # if (str_labels):
    #     plt.xticks(np.arange(len(all_param))[1:],labels=x_lables)
    plt.xticks(np.arange(len(amount_in_param)), labels=x_lables)
    plt.title("Amount of cases by "+param)
    plt.xlabel(param)
    plt.ylabel("amount of cases")
    plt.show()


def plot_amount_of_param_in_param(db, different_plots_data, y_data = None, should_revers = False, designated_labels = None,
                                  should_revers_x_labels = False, bar_plot = False, add_a_total = False, designated_x_labels = False):
    #get unique values in col:
    unique_vals_x = list(Counter(db[different_plots_data]).keys())
    x_labels = []
    sum_vals = []
    # if add_a_total == True:
    #     total_dict = {}
    # x_labels = set()
    for i, value in enumerate(unique_vals_x):
        # if (i < 7):

        if value != "-1":
            temp_db = db.loc[db[different_plots_data] == unique_vals_x[i]]
            if y_data == YEAR:
                temp_db = temp_db.loc[temp_db[y_data] >= 2000]

            # if type(db[y_data][0]) != str: #TODO - SORT ALL BY THE SAME VALUES!!!!!!!! add a counter and an is str varaA
            temp_db.reset_index(inplace = True)
            temp_db = temp_db.sort_values(by = y_data)


            unique_vals_y = list(Counter(temp_db[y_data]).keys())
            # print("UV_y = ", unique_vals_y)
            sum_vals_y = list(Counter(temp_db[y_data]).values())
            for x in unique_vals_y:
                if should_revers_x_labels:
                    if x[::-1] not in x_labels:
                    # x_labels.add(x)
                        x_labels.append(x[::-1])
                    # print(x[::-1])
                else:
                    if x not in x_labels:
                        x_labels.append(x)
            if add_a_total:
                sum_vals_y.append(sum_vals_y)
                """all_keys = total_dict.keys()
                for i, y in enumerate(sum_vals_y):
                    if unique_vals_y[i] in all_keys:
                        total_dict[unique_vals_y[i]] = sum_vals_y
                    else:
                        total_dict[unique_vals_y[i]] = sum_vals_y"""
            # if should_revers_x_labels:
            #     x_labels.append(value[::-1])
            # else:
            #     x_labels.append(value)
            # print("SV_y = ", sum_vals_y)
            # plt.scatter(unique_vals_y, sum_vals_y,label = value[::-1])

            if bar_plot:
                if designated_labels == None:
                    if should_revers:
                        plt.bar(unique_vals_y, sum_vals_y, alpha=0.5, label=value[::-1])
                    else:
                        plt.bar(unique_vals_y, sum_vals_y, alpha=0.5, label=value)

                else:
                    plt.bar(unique_vals_y, sum_vals_y, alpha=0.5, label=designated_labels[i])
            else:
                if (len(unique_vals_y) > 10):
                #     x_new = np.linspace(0, 15, 2000)
                #     y_new = interp1d(np.arange(len(unique_vals_y)), sum_vals_y)
                #     # a_BSpline = interp.make_interp_spline(np.arange(len(unique_vals_y)), sum_vals_y)
                #     # y_new = a_BSpline(x_new)
                #     xnew = np.linspace(0, 15, num=41, endpoint=True)
                #     plt.plot(x_new, y_new(x_new), label = value[::-1])
                #     coefficient_of_dermination = r2_score(sum_vals_y, y_new(np.linspace(0, 15, len(sum_vals_y))))
                #     print("for dist ",value," the r^2 val is: ",coefficient_of_dermination)
                # plt.legend(loc = "best")
                # plt.show()
                    if designated_labels == None:
                        if should_revers:
                            plt.plot(unique_vals_y, sum_vals_y,alpha = 0.5,label = value[::-1])
                        else:
                            plt.plot(unique_vals_y, sum_vals_y,alpha = 0.5,label = value)
                    else:
                        plt.plot(unique_vals_y, sum_vals_y, alpha=0.5, label=designated_labels[i])

    # x_labels = np.sort(list(x_labels))
    if type(x_labels) == str:
        plt.xticks(np.arange(len(list(x_labels))), labels = x_labels)
        # pass
    else:
        plt.xticks(np.arange(len(list(x_labels))), labels = x_labels)
        # pass
        # plt.xticks()
    print("x_labels = ",x_labels)
    plt.legend(loc = "best")

    if add_a_total:
        total = sum(sum_vals_y)
        plt.plot(np.arange(len(total)), total)

    if designated_x_labels:
        plt.xticks(np.arange(len(x_labels)), labels=designated_labels)

    if y_data == YEAR:
        plt.xlim(2000, 2020)
        plt.xticks(np.arange(2000, 2020), labels = x_labels)

    else:
        plt.xlim(0, len(x_labels))

    # plt.xlim(0,7)
    plt.title("The amount of cases per " + str(different_plots_data) + " per " + str(y_data))
    plt.xlabel(y_data)
    plt.ylabel("amount of cases")
    plt.show()
    # x_final_data = x_data
    # if type(x_data[0]) is str:
    #     m_dict = convert_str_to_int_dict(x_data)
    #     x_final_data = [m_dict[x_data[i]] for i in range(len(x_data))]
    # print("x_final_data = ",x_final_data)
    #find years:
    # all_yeras = Counter(y_data).keys()
    # for yr in all_yeras:
    #     count_by_years =
    #
    # for ftr,appr in Counter(x_data):
    #     temp = y_data[]
    #     plt.plot(x_final_data, )
    # from_search_to_local()

#-------------------- main ---------------------#
district_dict = {}
county_dict = {}
gzar_list = []
verdicts_list = []
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

    # for d in district_dict:
    #     print("key = ",d, "values = ",district_dict[d])
    #
    fromVerdictsToDB()
    print("no year found = ",counter_noYearFound)
    print("no accused name found = ",no_accusedName)
    print("no district found = ",no_districtCounter)
    print("no compensation found = ",no_compensCounter)
    print("no charges found", no_chargesCounter)
    print("no age found = ", no_ageCounter)
    # from_search_to_local()
    df = pd.read_csv("verdict_penalty.csv", error_bad_lines= False)
    df = df.loc[df['VOTED TIME'] != 0]
    # df = df.loc[df['VOTED TIME'] > 30]
    df = df.loc[df['VOTED TIME'] < 30]
    print(df['case_num'])
    print(np.median(df['VOTED TIME']))
    print(len(df['VOTED TIME']))
    plt.boxplot(df['VOTED TIME'])
    plt.title("התפלגות כלל העונשים עבור סעיפים ..."[::-1])
    plt.xlabel("סעיפי ענישה"[::-1])
    plt.xticks([1],labels=["345 - 351"])
    plt.ylabel("שנים"[::-1])
    plt.show()

    # plot_amount_of_param_in_param(df, IS_MINOR, DISTRICT,bar_plot= True, should_revers_x_labels= True ,designated_labels=["ןיטק","אל ןיטק","אל עודי"])#, add_a_total=True)
    # plot_amount_of_param_in_param(df, ASSULTED_GENDER, YEAR, bar_plot= True) # ,designated_labels=["ןיטק","אל ןיטק","אל עודי"])#, add_a_total=True)
    # plot_amount_of_param_in_param(df, IS_ANONYMOUS, YEAR,designated_labels = ["ינולפ","םש שי"])
    # plot_amount_per_param(df, JUDGE_NUM,bar_plot=True)#,str_labels= True, should_revers=True)



# ------------------------ Demo plots -----------------------------#
def demo_plot1():
    cities = ["Tel Aviv", "Jerusalem","Haifa","Be'er Sheva", "Nazareth"]
    trend = [-1, -1, 0, 0, 0, 10, 10, 10, 20,20]#, 2, 6, 6, 6, 6, 7, 7, 7, 7, 8, 9]
    for city in cities:
        years = np.arange(10)+2007
        compensations = np.random.randint(10, 100, size = 10)
        compensations = compensations + trend
        plt.plot(years, compensations, alpha=0.1,label = city)

        if city == "Be'er Sheva" or city == "Tel Aviv":
            if city == "Be'er Sheva":
                compensations = compensations - 15

            plt.plot(years, compensations,label = city)
        #plt.scatter(years, compensations, label = city)


    plt.title ("Avarege compensation amount across years according to district")
    plt.xlabel("Year")
    plt.ylabel("amont in thousand Shekels")
    plt.xticks(years)
    plt.legend(loc = "best")
    plt.savefig("try2_trend")
    plt.show()

def demo_plot_2():
    compensations = np.random.randint(10, 100, size=4)
    # female_majority = np.random.randint(0,3, size=20)
    print(compensations)
    for i in range(len(compensations)):
        if i < 2:
            plt.bar([i*10], [compensations[i]*10], color = "cornflowerblue" )
        else:
            plt.bar([i*10], [compensations[i]*10], color = "lightcoral" )
    plt.xticks (np.arange(4)*10)
    plt.title("The average compensation amount as a factor of\n the amount of female judges")
    plt.xlabel("The amount of female judges (X10)")

    plt.show()

def demo_plot_4():
    judge = np.array([ 1,    2,   3,   2,    1    , 3,   3,   3,   3, 3,  1,    3,   3,   3,   3,  3, 2])
    # judge = np.array([3,  1,    2,   3,   3,   2,    1    , 3,   3,   3,   3, 3,  1,    3,   3,   3,   3,  3, 2])
    lines = np.array([ 130, 458, 665, 1005, 123   ,824, 63, 251, 174, 76, 116, 972, 288, 579, 109, 1792, 295])
    # lines = np.array([19, 130, 458, 665,10468, 1005, 123   ,824, 63, 251, 174, 76, 116, 972, 288, 579, 109, 1792, 295])

    for i in range(3):
        plt.bar([i+1], [np.average(lines[np.argwhere(judge == i+1)])])
    plt.xticks([1,2,3])
    plt.xlabel("number of judges")
    plt.ylabel("number of sentences")
    plt.title("Average amount of sentences as factor\n of amount of judges")
    plt.show()


# ["https://www.nevo.co.il/psika_html/shalom/SH-96-84-HK.htm" - 1998,
#         "https://www.nevo.co.il/psika_html/mechozi/m06000511-a.htm" - 2006,
#         "https://www.nevo.co.il/psika_html/mechozi/ME-09-02-10574-380.htm" - 2009,
#         "https://www.nevo.co.il/psika_html/mechozi/m06007004-660.htm" - 2006,
#         "https://www.nevo.co.il/psika_html/mechozi/m06020001.htm" - 2006,
#         "https://www.nevo.co.il/psika_html/shalom/s01003122-438.htm" - 2004,
#         "https://www.nevo.co.il/psika_html/shalom/s981928.htm" - 1999,
#         "https://www.nevo.co.il/psika_html/mechozi/m011190a.htm" - 2000,
#         "https://www.nevo.co.il/psika_html/mechozi/m01000232.htm" - 2005,
#         "https://www.nevo.co.il/psika_html/mechozi/m99934.htm" - 2000,
#         "https://www.nevo.co.il/psika_html/mechozi/ME-12-01-13327-55.htm" - 2015,
#         "https://www.nevo.co.il/psika_html/mechozi/me-93-76-a.htm" - 1994,
#         "https://www.nevo.co.il/psika_html/mechozi/m01171.htm" - 2001,
#         "https://www.nevo.co.il/psika_html/mechozi/ME-16-12-8398-11.htm" - 2016,
#         "https://www.nevo.co.il/psika_html/mechozi/m01000405-a.htm" - 2005,
#         "https://www.nevo.co.il/psika_html/mechozi/m00001039-148.htm" - 2001,
#         "https://www.nevo.co.il/psika_html/mechozi/m001129.htm" - 2001,
#         "https://www.nevo.co.il/psika_html/mechozi/m00000935-103.htm" - 2001,
#         "https://www.nevo.co.il/psika_html/mechozi/ME-98-4124-HK.htm" - 2003
#         ]
