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
import datefinder

VERDICTS_DIR = "verdicts/"

DISTRICT = "district"
CASE_NUM = "case_num"
ACCUSED_NAME = "accused"
NUM_LINES = "num_lines"
YEAR = "year"
COMPENSATION = "compensation"
ACCUSED = "accused"
ARCHA = "archa"
JuDGE_NUM = ""
VIRGINS = ""

YEAR_REG = r"([0-3]{0,1}[0-9]\/[0-2]{0,1}[0-9]\/[0-2]{0,1}[0-9]{1,3})|([0-3]{0,1}[0-9]\.[0-2]{0,1}[0-9]\.[0-2]{0,1}[0-9]{1,3})"

no_districtCounter = 0
no_caseNameCounter = 0
no_chargesCounter = 0
no_compensCounter = 0
no_accusedName = 0

#### CONVENTIONS
# All the extraction functions RETURN the value at the end, which will then be submitted into the DB
# If an extraction function doesn't find the parameter, we return -1

################# Generic Functions For Extraction ######################

def allTheTextAfterAWord(text, word, until=-1):
    targetStartIndex = text.find(word)
    return text[targetStartIndex:until]


def findBetweenParentheses(text):
    targetStartIndex = text.find("(") + 1
    parenthesisAfterEndIndex = text.find(")", targetStartIndex)
    return text[targetStartIndex:parenthesisAfterEndIndex]


def extractWordAfterKeywords(text, words):
    for word in words:
        keywordIndex = text.find(word)
        if keywordIndex != -1:
            break
    if keywordIndex == -1:
        return -1
    targetStartIndex = keywordIndex + len(word) + 1  # plus one because of space
    spaceAfterEndIndex = text.find(" ", targetStartIndex)
    return text[targetStartIndex:spaceAfterEndIndex]

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
                    if start + i < num_lines:
                        relevant_lines.append(t_by_lines[start + i])
                # print("\n".join(relevant_lines))
                return "\n".join(relevant_lines)
            else:
                return ""
    return -1

##############################PARAMETERS#######################################


def accusedName(text):
    accused_found = extractWordAfterKeywords(text, ["נ'"])
    return accused_found

def compensation(text):  # TODO: in one instance finds the salary instead of compensation
    # text = allTheTextAfterAWord(text, "סיכום") # TODO: not always a title of "summary"
    return extractWordAfterKeywords(text, ["סך של כ-", "סך של"]) # TODO: sometimes there's only "סך"

def courtArea(text):
    return findBetweenParentheses(text)

def howManyLines(text):
    return len(text.split("."))

def ageOfVictim(text):
    found = extractWordAfterKeywords(text, [" כבת", " בת"])
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


############ Main Functions ################
def extractParameters(text, db, year, case_name):
    #TODO : figure out how to limit the search area (ideas - number of lines, not in entioned laws, before discausion etc...)
    # think of a good structure to call each function of extraction and put the output in the correct column
    accused_name = accusedName(text)
    db = db.append({'accused_name': accused_name}, ignore_index=True)
    if accused_name == -1:
        global no_accusedName
        no_accusedName += 1

    charges =  extractLaw(text)
    db = db.append({'charges':charges},ignore_index=True)
    if charges == -1:
        global no_chargesCounter
        no_chargesCounter += 1

    compens = compensation(text)
    db = db.append({'compensation':compens},ignore_index=True)
    if compens == -1:
        global no_compensCounter
        no_compensCounter += 1

    district = courtArea(text)
    db = db.append({"district": district},ignore_index=True)
    if district == -1:
        global  no_districtCounter
        no_districtCounter +=1

    lines_num = howManyLines(text)
    db = db.append({"lines_num": lines_num},ignore_index=True)
    db = db.append({"year": year},ignore_index=True)
    db = db.append({"case_name": case_name},ignore_index=True)
    interestingWords(text)
    # ['case_num', 'year', 'district', 'charges', 'accused_name', 'lines_num'])
    case_ftr = pd.DataFrame([[case_name,   year,   district,  charges, compens,           accused_name ,lines_num]],
                            columns=['case_num', 'year', 'district', 'charges', 'compensation','accused_name', 'lines_num'])
    # db = pd..appended(case_ftr)
    return case_ftr

def createNewDB():
    # create a xls file with the right columns as the parameters
    # TODO: idea to use a dictionary as the structure to create this DB
    df = pd.DataFrame(columns=[CASE_NUM, YEAR, DISTRICT, 'charges', COMPENSATION, ACCUSED, NUM_LINES])
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


def plot_amount_of_param_in_param(db, col_name, y_data = None):
    #get unique values in col:
    unique_vals_x = list(Counter(db[col_name]).keys())
    to_plot = []
    for value in unique_vals_x:
        temp_db = db.loc[db[col_name] == value]
        unique_vals_y = list(Counter(temp_db[y_data]).keys())
        sum_vals_y = list(Counter(temp_db[y_data]).values())
        # print("UV_y = ", unique_vals_y)
        # print("SV_y = ", sum_vals_y)
        plt.scatter(unique_vals_y, sum_vals_y,label = value[::-1])
        plt.plot(unique_vals_y, sum_vals_y,alpha = 0.1)
    plt.legend(loc = "best")
    plt.xlim(1985,2025)
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

def extractLaw(text):
    reg_word = "חקיקה שאוזכרה"
    # reg_word = "פסק ה*דין"
    relevant = get_lines_after(text, reg_word, 10, 3)
    all_charges = []
    amount_appeard = []
    line_appeadr =[] #TODO Deduce from the line all apeared in which is previuously mentioned
    for chrg in CHARGES:
        if relevant != -1:
            if relevant.find(chrg) != -1:
                all_charges.append([chrg, text.count(chrg)])
                amount_appeard.append(text.count(chrg))
    print("section = ", all_charges)
    # print("appeared: ", amount_appeard)
    if len(all_charges) > 0:
        return all_charges
    else:
        return -1

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
        year = date.split(".")[2]
        return (int)(year)
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
    case_names = []
    compens = []
    districts = []
    charges = []
    lines_number = []
    all_accused = []

    directory = VERDICTS_DIR               #text files eddition:
    years = []
    # years = [1998, 2006, 2009, 2006, 2006, 2004, 1999, 2000, 2005, 2000, 2015, 1994, 2001, 2016, 2005, 2001, 2001, 2001, 2003]
    counter = -1
    # print("years len = ", len(years))
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".txt"):
            counter +=1
            file_name = os.path.join(directory, filename)
            text = open(file_name, "r", encoding="utf-8").read()
            years.append(extract_publish_dates(text))
            print("^^^ File is ", file_name, " ^^^")
            print("filename = ",filename,"counter = ",counter,"year = ",years[counter])
            db = pd.concat([db, extractParameters(text, db, years[counter], filename)])

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
CHARGES = ['345', '346', '347', '348', '349', '350', '351']

# fromVerdictsToDB(urls)

searches_results = ["search11.txt","search12.txt","search13.txt","search14.txt","search15.txt","search16.txt","search17.txt","search18.txt","search19.txt","search20.txt"]

def from_search_to_local():
    dir = "SearchResults\\"
    for serach in searches_results:
        allURLS = get_urls_from_text_source(dir+serach)
        for url in allURLS:
            text = urlToText(url)
            print("url = ",url)
            if url.find("mechozi") > 0:
                 add_to_txt_db(url,text,"mechozi")
            elif url.find("shalom") > 0:
                add_to_txt_db(url, text, "shalom")
            else:
                print("didn't work for: ",url)

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

if __name__ == "__main__":
    fromVerdictsToDB()
    print("no year found = ",counter_noYearFound)
    print("no accused name found = ",no_accusedName)
    print("no district found = ",no_districtCounter)
    print("no compensation found = ",no_compensCounter)
    print("no charges found", no_chargesCounter)
    # from_search_to_local()
    # df = pd.read_csv("out4.csv", error_bad_lines= False)
    # plot_amount_of_param_in_param(df,DISTRICT,YEAR)


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

#------------------ Real plots ---------------------------------#
def plot_year_and_dist(batch):
    batch = batch.loc[batch[YEAR] != '-1']
    all_years = batch[YEAR]
    all_dist = batch[DISTRICT]

    plt.plot(all_years,all_dist)
