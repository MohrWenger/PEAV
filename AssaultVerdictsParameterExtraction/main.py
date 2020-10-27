from bs4 import BeautifulSoup
import urllib.request

def nameExtract(file):
    pass


def ExtractParameters(text, db):
    # think of a good structure to call each function of extraction and put the output in the correct column
    pass


def createNewDB():
    # create a xls file with the right columns as the parameters
    # TODO: idea to use a dictionary as the structure to create this DB
    pass

def extractLaw():
    pass

def urlToText(url):
    webUrl = urllib.request.urlopen("https://www.nevo.co.il/psika_html/elyon/05006450-o08-e.htm")
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
    return text
"""
This function receives a path with many verdicts (presumably in word or html format), and uses the code to create a
database
"""
def fromVerdictsToDB(url):
    # call createNewDB
    # go through files in a loop
        # for each file, call ExtractParameters
    text = urlToText(url)
