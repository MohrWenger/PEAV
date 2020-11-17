from bs4 import BeautifulSoup
import urllib.request
# import hebpipe
import os
import re
import string
import matplotlib as plt
import numpy as np
from collections import Counter


VERDICTS_DIR = "verdicts/"

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
    if limit == "eof":
        limit = len(t_by_lines)
    start = 0
    for line in range(len(t_by_lines)):
        if re.search(word, t_by_lines[line]):
            start = line + startAfter
            relevant_lines = []
            if start < limit:
                for i in range(amount):
                    relevant_lines.append(t_by_lines[start + i])
                # print("\n".join(relevant_lines))
                return "\n".join(relevant_lines)
            else:
                return ""
    return -1

##############################PARAMETERS#######################################


def accusedName(text):
    return extractWordAfterKeywords(text, ["נ'"])


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


def extractParameters(text, db):
    #TODO : figure out how to limit the search area (ideas - number of lines, not in entioned laws, before discausion etc...)
    # think of a good structure to call each function of extraction and put the output in the correct column
    # print(accusedName(text))
    # extractLaw(text)
    print("com[ = ",compensation(text))
    print(courtArea(text))
    print(howManyLines(text))
    interestingWords(text)
    return accusedName(text)


def createNewDB():
    # create a xls file with the right columns as the parameters
    # TODO: idea to use a dictionary as the structure to create this DB
    return dict()


def urlToText(url):
    webUrl = urllib.request.urlopen(url)
    html = webUrl.read()
    soup = BeautifulSoup(html, features="html.parser")

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
    # print(text)
    return text


#################### mohr current working on ###############################33
def convert_str_to_int_dict(str_arr):
        unique_vals = list(Counter(str_arr).keys())
        print(unique_vals)
        int_vals = np.arange(len(unique_vals))
        new_dict = {unique_vals[i]: int_vals[i] for i in range(len(unique_vals))}
        print(new_dict)
        return  new_dict


def generic_plot_func(x_data, y_data = None):
    x_final_data = x_data
    if type(x_data[0]) is str:
        m_dict = convert_str_to_int_dict(x_data)
        x_final_data = [m_dict[x_data[i]] for i in range(len(x_data))]
    print("x_final_data = ",x_final_data)
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
        if relevant.find(chrg) != -1:
            all_charges.append([chrg, text.count(chrg)])
            amount_appeard.append(text.count(chrg))
    print("section = ", all_charges)
    # print("appeared: ", amount_appeard)
    return all_charges


def add_to_txt_db(url, text, court_type):
    name_file = url.strip("https://www.nevo.co.il/psika_html/"+court_type+"/")
    with open(VERDICTS_DIR + name_file + ".txt", "w", encoding="utf-8") as newFile:
        newFile.write(text)


"""
This function receives a path with many verdicts (presumably in word or html format), and uses the code to create a
database
"""
def fromVerdictsToDB(urls):
    db = createNewDB()
    all_accused = []

    directory = VERDICTS_DIR               #text files eddition:
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_name = os.path.join(directory, filename)
            text = open(file_name, "r", encoding="utf-8").read()
            print("^^^ File is ", file_name, " ^^^")
            param1 = extractParameters(text, db)
            all_accused.append(param1)
            generic_plot_func(all_accused)
        else:
            continue

    # for url in urls:                       #html edition
    #     text = urlToText(url)
    #     print(url)  # as kind of a title
    #     ExtractParameters(text, db)
        # add_to_txt_db(url, text,"mechozi")

        years = [1998, 2006, 2009, 2006, 2006, 2004, 1999, 2000, 2015, 1994, 2001, 2016, 2005, 2001, 2001, 2001, 2003]
        comp = [-1, -1, -1, 10000, 3500, -1, -1, 7000, -1, -1, -1, 3000, -1, 170000, 75000, -1, -1, -1, -1]
        places = ["שלום תל אביב-יפו", "מחוזי נצ'", "מחוזי מרכז",  "מחוזי י-ם", "שלום ירושלים", "פורסם בנבו, 10.10.1999",
                  "מחוזי תל אביב-יפו", "מחוזי חיפה", "מחוזי באר שבע", "מחוזי נצרת" , "מחוזי ב\"ש", "מחוזי חיפה",
                  "מחוזי נצרת", "מחוזי תל אביב-יפו", "מחוזי חיפה", "מחוזי תל אביב-יפו", "מחוזי תל אביב-יפו", "מחוזי באר שבע", "מחוזי תל אביב-יפו"]
        generic_plot_func(places, years)

        print("\n\n")


urls = []

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

CHARGES = ['345', '346', '347', '348', '349', '350', '351']

fromVerdictsToDB(urls)
