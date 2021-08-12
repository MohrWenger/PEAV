import json
import os
import re
import string
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
from bs4 import BeautifulSoup
# import requests_html



import urllib.request as urlb
import urllib3

# VERDICTS_DIR = r"C:\Users\oryiz\PycharmProjects\PEAV\AssaultVerdictsParameterExtraction\verdicts"
# VERDICTS_DIR = "verdicts/" #TODO this is the directory the files are going to be saved to
VERDICTS_DIR = "/Users/tomkalir/Downloads/final_verdicts_dir/" #TODO this is the directory the files are going to be saved to
NEW_VERDICTS_DIR = "final_verdicts_dir/"
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
HTML_FILES = "קובץ"

def urlToText(url):
    """
    This function extracts the text of the URL and returns it as string
    :param url: The URL to read
    :return: The text of the URL as a long string
    """


    print("url = ",url)

    # webUrl = urllib.request.urlopen("file\\:"+url)
    # html = webUrl.read()
    # r = urlb.urlopen(url)
    # x = BeautifulSoup.BeautifulSoup(r.read)
    # r.close()


    html = open(url,  errors="ignore").read()
    webUrl = urlb.urlopen("file:///"+url)
    html = webUrl.read()
    # soup = BeautifulSoup(html, features="html.parser", from_encoding= 'utf-8')
    # soup = BeautifulSoup(html, features="html.parser", from_encoding='utf-8-sig')
    #possibly we need encoding: windows-1255
    soup = BeautifulSoup(html, features="html.parser", from_encoding='found_encoding')
    print(soup.original_encoding)
    soup.prettify('utf-8')

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


def add_to_txt_db(url, name_only = False):
    """
    This function saves a URL to a text file with the case number as the file name.
    :param url: The url to convert
    :param name_only: If True then it doesn't write the file but only returns the name.
    :return:The name of the file based on the given URL
    """
    text = urlToText(url)
    if text == "":
        print("breakpoint at ",url)
    name_file = url.split("/")[-1]
    name_file = url.split("\\")[-1]
    name_file = name_file.replace(".htm",".txt")

    if not name_only:
        with open(NEW_VERDICTS_DIR + name_file, "w", encoding="utf-8") as newFile:
            newFile.write(text)
    return name_file

def list_of_urls_to_local(list_of_urls):
    for i in range(len(list_of_urls)):
        if type(list_of_urls[i]) == str:
            print(add_to_txt_db(list_of_urls[i]))


# def fromVerdictsToDB(path_to_files):
#     """
#     This function receives a path with many verdicts (presumably in word or html format), and uses the code to create a
#     database
#     :return:
#     """
#     batch = pd.read_csv(path_to_files, error_bad_lines = False)
#     files = batch[HTML_FILES]
#     with open('test_case_filenames.txt') as json_file:
#         relevant_cases = json.load(json_file)
#     db = pd.DataFrame()
#     directory = VERDICTS_DIR               #text files eddition:
#     counter = -1
#     all = 0
#     not_good = 0
#     # for i in range(len(files)): #when iterateing through all files in igud colum
#     #     if type(files[i]) == str:
#     for i, filename in enumerate(os.listdir(directory)): #when iterateing through all files in folder
#         if filename.endswith(".txt") and filename in relevant_cases:
#
#             filename = add_to_txt_db(files[i])
#             counter += 1
#             file_name = os.path.join(directory, filename)
#             text = open(file_name, "r", encoding="utf-8").read()
#
#             print("^^^ File is ", file_name, " ^^^ - not psak")
#             all, not_good, main_penalty, sentence, all_sentences, all_times, time, time_unit = extracting_penalty(text, filename, all, not_good) #Here I call the penalty func
#             batch.loc[i,"PENALTY_SENTENCE"] = main_penalty
#             batch.loc[i,"VOTED TIME"] = time
#             sentence_line = pd.DataFrame([[file_name,"Gzar", main_penalty,     sentence,          all_sentences,   all_times,       time, time_unit]], #here I add values to DB
#                                          columns =[CASE_NUM,     "TYPE","Main Punishment","PENALTY_SENTENCE", "ALL SENTENCES", "OPTIONAL TIMES", "VOTED TIME", "units"]) #Here adding a title
#             db = pd.concat([db,sentence_line ])
#
#         else:
#             continue
#     db.to_csv('verdict_penalty.csv', encoding= 'utf-8') #writing the db to a csv file

district_dict = {}
county_dict = {}
gzar_list = []
verdicts_list = []

if __name__ == "__main__":

    directory = r"C:\Users\oryiz\Downloads\html"
    # directory = VERDICTS_DIR
    batch = pd.read_csv("db_csv_files/Igud_Gzar2 - Sheet1.csv", error_bad_lines = False)
    for i, filename in enumerate(os.listdir(directory)):
        # print(i)
        new_name = filename.strip("verdicts")
        print("new filename: ",  VERDICTS_DIR+new_name)
        f = open(NEW_VERDICTS_DIR + new_name, "w")
        origin_f = open(VERDICTS_DIR + filename)
        # print(origin_f.read())
        # f.write(origin_f.read())
        if filename.endswith(".htm"):
            add_to_txt_db(directory+"\\"+filename)

    # list_of_urls_to_local(batch[HTML_FILES])