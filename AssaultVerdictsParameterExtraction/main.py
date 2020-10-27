from bs4 import BeautifulSoup
import urllib.request


def extractWordAfterKeyword(text, word):
    keywordIndex = text.find(word)
    targetStartIndex = keywordIndex + len(word) + 1  # plus one because of space
    spaceAfterEndIndex = text.find(" ", targetStartIndex)
    return text[targetStartIndex:spaceAfterEndIndex]


def accusedName(text):
    return extractWordAfterKeyword(text, "נ'")


def ExtractParameters(text, db):
    # think of a good structure to call each function of extraction and put the output in the correct column
    print(accusedName(text))


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
        ExtractParameters(text, db)


urls = ["https://www.nevo.co.il/psika_html/shalom/SH-96-84-HK.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m06000511-a.htm",
        "https://www.nevo.co.il/psika_html/mechozi/ME-09-02-10574-380.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m06007004-660.htm",
        "https://www.nevo.co.il/psika_html/mechozi/m06020001.htm",
        "https://www.nevo.co.il/psika_html/shalom/s01003122-438.htm",
        "https://www.nevo.co.il/psika_html/shalom/s981928.htm"]

fromVerdictsToDB(urls)
