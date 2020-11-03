from bs4 import BeautifulSoup
import urllib.request


def allTheTextAfterAWord(text, word):
    targetStartIndex = text.find(word)
    return text[targetStartIndex:]


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

##############################PARAMETERS#######################################

def accusedName(text):
    return extractWordAfterKeywords(text, ["נ'"])


def compensation(text):  # TODO: in one instance finds the salary instead of compensation
    # text = allTheTextAfterAWord(text, "סיכום") # TODO: not always a title of "summary"
    return extractWordAfterKeywords(text, ["סך של כ-", "סך של"])

def courtArea(text):
    return findBetweenParentheses(text)

def ExtractParameters(text, db):
    #TODO : figure out how to limit the search area (ideas - number of lines, not in entioned laws, before discausion etc...)
    # think of a good structure to call each function of extraction and put the output in the correct column
    print(accusedName(text))
    extractLaw(text)
    print(compensation(text))
    print(courtArea(text))

def extractLaw(text):
    all_charges = []
    for chrg in CHARGES:
        if text.find(chrg) != -1:
            all_charges.append(chrg)
    print("section = ",all_charges)

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


def add_to_txt_db(url,text,court_type):
    name_file = url.strip("https://www.nevo.co.il/psika_html/"+court_type+"/")
    with open(
            "C:\\Users\\oryiz\\PycharmProjects\\PEAV\\AssaultVerdictsParameterExtraction\\verdicts\\" + name_file + ".txt",
            "w", encoding="utf-8") as newFile:
        newFile.write(text)


"""
This function receives a path with many verdicts (presumably in word or html format), and uses the code to create a
database
"""
def fromVerdictsToDB(urls):
    db = createNewDB()
    # go through files in a loop
        # for each file, call ExtractParameters
    for url in urls:
        text = urlToText(url)
        print(url)  # as kind of a title
        ExtractParameters(text, db)
        add_to_txt_db(url, text,"mechozi")
        print("\n\n")


urls = [#"https://www.nevo.co.il/psika_html/shalom/SH-96-84-HK.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m06000511-a.htm",
        "https://www.nevo.co.il/psika_html/mechozi/ME-09-02-10574-380.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m06007004-660.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m06020001.htm",
        # "https://www.nevo.co.il/psika_html/shalom/s01003122-438.htm",
        # "https://www.nevo.co.il/psika_html/shalom/s981928.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m011190a.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m01000232.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m99934.htm",
        "https://www.nevo.co.il/psika_html/mechozi/ME-12-01-13327-55.htm",
        "https://www.nevo.co.il/psika_html/mechozi/me-93-76-a.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m01171.htm",
        "https://www.nevo.co.il/psika_html/mechozi/ME-16-12-8398-11.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m01000405-a.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m011314.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m00001039-148.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m001129.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m00000935-103.htm",
        "https://www.nevo.co.il/psika_html/mechozi/ME-98-4124-HK.htm"
        ]

print("we are now masters of git")
CHARGES = ['345','346','347','348','349','350','351']

fromVerdictsToDB(urls)
