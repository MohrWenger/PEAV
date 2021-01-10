from bs4 import BeautifulSoup
import urllib.request
# import hebpipe
import os
import re
import string
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
import json
import datefinder

VERDICTS_DIR = "verdicts/"

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

YEAR_REG = r"([0-3]{0,1}[0-9]\/[0-2]{0,1}[0-9]\/[0-2]{0,1}[0-9]{1,3})|([0-3]{0,1}[0-9]\.[0-2]{0,1}[0-9]\.[0-2]{0,1}[0-9]{1,3})"

C345 = '345'
C346 = '346'
C347 = '347'
C348 = '348'
C349 = '349'
C350 = '350'
C351 = '351'

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
    print("relevant = ",relevant )
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
    print(district_dict)

    print("court = ", court)

    for dist in district_dict:
        print("districts = ", district_dict[dist])
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

    elif accused_name == "פלוני":
        isAnonymous = True

    if (accused_name != "מדינת"):
        minor = is_minor(text)
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
            print("found dist - ", district)
            if district == -1:
                global no_districtCounter
                no_districtCounter += 1
                district = court.split(" ")[1:][0]
                print("not found dist - ",district)
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
        case_ftr = pd.DataFrame([[case_name, day,month,  year,   court,  district, level, county ,age, minor ,compens, accused_name, isAnonymous,
                                  assultedGender, judges_amount, female_J, male_J,sexPerc, lines_num,
                                  charges[0],charges[1],charges[2],charges[3],charges[4],charges[5],charges[6]]],

                                columns=['case_num',DAY, MONTH, YEAR, 'court',DISTRICT,'level','county',AGE, IS_MINOR ,'compensation','accused_name', IS_ANONYMOUS,
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
    print(soup.original_encoding)
    print(text)
    return text

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
    print("name = ",name)
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
        return True if np.sum(results) > 0 else False
    else:
        return -1


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

def add_to_txt_db(url, text, court_type):
    name_file = url.strip("https://www.nevo.co.il/psika_html/"+court_type+"/")
    with open(VERDICTS_DIR + name_file + ".txt", "w", encoding="utf-8") as newFile:
        newFile.write(text)

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
    db = createNewDB()
    # case_names = []
    # compens = []
    # districts = []
    # charges = []
    # lines_number = []
    # all_accused = []

    directory = VERDICTS_DIR               #text files eddition:
    # years = []
    # years = [1998, 2006, 2009, 2006, 2006, 2004, 1999, 2000, 2005, 2000, 2015, 1994, 2001, 2016, 2005, 2001, 2001, 2001, 2003]
    counter = -1
    # print("years len = ", len(years))
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".txt"):
            counter +=1
            file_name = os.path.join(directory, filename)
            text = open(file_name, "r", encoding="utf-8").read()

            print("^^^ File is ", file_name, " ^^^")
            # print("filename = ",filename,"counter = ",counter,"year = ",years[counter])
            verd_line = extractParameters(text, db, filename)
            if verd_line is not None:
                db = pd.concat([db,verd_line ])

            # all_accused.append( accusedName(text))
            # #charges.append(extractLaw(text))
            # compens.append(compensation(text))
            # districts.append( courtArea(text))
            # lines_number.append(howManyLines(text))
            # case_names.append(filename)
        else:
            continue
    db.to_csv('out4.csv', encoding= 'utf-8')
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
                                  should_revers_x_labels = False, bar_plot = False, add_a_total = False):
    #get unique values in col:
    unique_vals_x = list(Counter(db[different_plots_data]).keys())
    x_labels = []
    sum_vals = []
    # if add_a_total == True:
    #     total_dict = {}

    for i, value in enumerate(unique_vals_x):
        # if (i < 7):

        if value != "-1":
            temp_db = db.loc[db[different_plots_data] == unique_vals_x[i]]
            temp_db = temp_db.loc[temp_db[y_data] != -1]

            if type(db[y_data][0]) != str: #TODO - SORT ALL BY THE SAME VALUES!!!!!!!!
                temp_db.reset_index(inplace=True)
                temp_db = temp_db.sort_values(by= y_data)

            sum_vals_y = list(Counter(temp_db[y_data]).values())
            unique_vals_y = list(Counter(temp_db[y_data]).keys())
            print("UV_y = ", unique_vals_y)

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
            if should_revers_x_labels:
                for x in unique_vals_y:
                    x_labels.append(x[::-1])
            if bar_plot:
                if designated_labels == None:
                    if should_revers:
                        plt.bar(unique_vals_y, sum_vals_y, alpha=0.5, label=value[::-1])
                    else:
                        plt.bar(unique_vals_y, sum_vals_y, alpha=0.5, label=value)

                else:
                    plt.bar(unique_vals_y, sum_vals_y, alpha=0.5, label=designated_labels[i])
            else:
                if designated_labels == None:
                    if should_revers:
                        plt.plot(unique_vals_y, sum_vals_y,alpha = 0.5,label = value[::-1])
                    else:
                        plt.plot(unique_vals_y, sum_vals_y,alpha = 0.5,label = value)
                else:
                    plt.plot(unique_vals_y, sum_vals_y, alpha=0.5, label=designated_labels[i])
        if type(x_labels[0]) == str:
            plt.xticks(np.arange(len(x_labels)), labels=x_labels)
        else:
            plt.xticks(x_labels)

    plt.legend(loc = "best")
    if add_a_total:
        total = sum(sum_vals_y)
        plt.plot(np.arange(len(total)), total)

    if designated_labels:
        plt.xticks(np.arange(len(x_labels)), labels=designated_labels)

    if y_data == YEAR:
        plt.xlim(2000, 2020)

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
    pass
    # from_search_to_local()

#-------------------- main ---------------------#
district_dict = {}
county_dict = {}
if __name__ == "__main__":

    # district_dict = {}
    # with open('data.txt') as json_file:
    #     district_dict = json.load(json_file)
    #
    # with open('county_list.txt') as json_file:
    #     county_dict = json.load(json_file)
    #
    # for d in district_dict:
    #     print("key = ",d, "values = ",district_dict[d])
    #
    # fromVerdictsToDB()
    print("no year found = ",counter_noYearFound)
    print("no accused name found = ",no_accusedName)
    print("no district found = ",no_districtCounter)
    print("no compensation found = ",no_compensCounter)
    print("no charges found", no_chargesCounter)
    print("no age found = ", no_ageCounter)
    # from_search_to_local()
    df = pd.read_csv("out4.csv", error_bad_lines= False)
    print(len(df))

    plot_amount_of_param_in_param(df, IS_MINOR, "county",bar_plot=True, should_revers_x_labels=True)#, designated_labels=["FALSE","TRUE"])#, add_a_total=True)
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
