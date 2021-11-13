import pandas as pd
from AssaultVerdictsParameterExtraction import penalty_extraction as pe
import numpy as np
import re
import matplotlib.pyplot as plt

SENTENCE = "sentence"
FILE_NAME = "filename"
VOTED_TIME = "VOTED TIME"
YEAR_REG = r"([0-3]{0,1}[0-9]\/[0-2]{0,1}[0-9]\/[0-2]{0,1}[0-9]{1,3})|([0-3]{0,1}[0-9]\.[0-2]{0,1}[0-9]\.[0-2]{0,1}[0-9]{1,3})"

def extracted_time_db(db):
    sentence_list = db[SENTENCE].to_list()
    case_name_list = db[FILE_NAME].to_list()
    time_db = pe.from_sentence_list(case_name_list, sentence_list)
    return time_db

def extract_publish_dates(text):
    # t = get_lines_after(text, "נ'|נגד", 50,0) TODO - when the year is in paranthesis
    # name = text.splitlines()[0].replace("(","")
    # name = name.replace(")","")
    # print("name = ",name)
    name = text
    #print("ext = ", extractWordAfterKeywords(name, " בנבו, "))
    # matches = re.findall(YEAR_REG,name)
    matches = re.findall('[12][890][0-9][0-9]',name)
    for match in matches:
        if 1990 < int(match) < 2021:
            print("matches = ",match)
            return int(match)
        # date = matches[0]
        # print("date = ",date)
        # day, month, year = date.split(".")
        # return [(int)(day),(int)(month),(int)(year)]
    else:
        print("check here cause got -1")
        return "-1"

def add_years(db):
    file_names = db[FILE_NAME]
    years = []
    print(len(file_names))
    nan_files_counter = 0
    for i,f in enumerate(file_names):
        print("f = ", f)
        if f != f:
            nan_files_counter += 1
            years.append(-1)
        else:
            name = "D:\\PEAV\\AssaultVerdictsParameterExtraction\\final_verdicts_dir\\"+f
            text = open(name, encoding='utf-8').read()
            years.append(extract_publish_dates(text))
    db["years"] = years
    print(i)
    db.to_csv("times and years.csv", encoding = "utf-8")

def values_by_col(db, col):
    values = np.sort(np.unique(db[col]))
    median_time = []
    number_of_cases_hist =[]
    for i,val in enumerate(values):
        temp_db = db.loc[db[col]==val]
        times = [int(x) for x in temp_db[VOTED_TIME].to_list()]
        median_time.append(np.median(times))
        print("For ",val," there are ",len(temp_db),"cases")
        number_of_cases_hist.append(len(temp_db))

    number_of_cases_hist = np.array(number_of_cases_hist)

    if col == "years":
        fig, ax = plt.subplots(figsize=(10, 10 * 2 / 3))
        # plt.title("The Sentenced Imprisonment By Years\n(1178) cases extracted by RB")
        plt.plot(values[number_of_cases_hist > 20], median_time[number_of_cases_hist > 20])
        plt.show()
        #
        # fig.suptitle("Number of Cases By Years\n",fontsize = 25)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(3)
        plt.xlabel("Years", fontsize = 18)
        plt.ylabel("Number of Cases", fontsize = 18)
        plt.bar(values,number_of_cases_hist)
        plt.xticks( fontsize = 15) #להוסיף עוד בציר X
        plt.yticks( fontsize = 15) #להוסיף עוד בציר X
        plt.show()

    elif col == VOTED_TIME:
        fig, ax = plt.subplots(figsize=(10, 10 * 2 / 3))

        # fig.suptitle("Histogram of Months For Actual Imprisonment\n"
        #              "Based on performance of the rule based model\n\n", fontsize=22)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        plt.xlabel("Months Sentenced", fontsize=18)
        plt.ylabel("Number of Cases", fontsize=18)
        # plt.bar(values,number_of_cases_hist) #Widen the bins
        hist = np.histogram(db[VOTED_TIME],bins=20)
        # hist_smothed = np.histogram(db[VOTED_TIME],bins=10)
        print(hist[1])
        plt.hist(db[VOTED_TIME],bins=hist[1]+1) #Widen the bins
        # plt.plot(hist[1][:20],hist[0])
        print("number of values = ", len(values))
        print("till 15:", np.sum(number_of_cases_hist[:16]) / np.sum(number_of_cases_hist))
        print("Most common:", np.max(number_of_cases_hist))
        # plt.plot(np.repeat(12,30),np.arange(30), color = 'black', linestyle = "--") #Widen the bins
        plt.xticks( hist[1][::2], fontsize=15)  # להוסיף עוד בציר X
        plt.yticks(fontsize=15)  # להוסיף עוד בציר X
        # plt.ylim(0,26)
        plt.show()
    # plt.title("Histogram of Months For Actual Imprisonment\nBased on performance of the rule based model")
    # plt.hist(number_of_cases_hist,bins=30) #Widen the bins
    # plt.xlabel("Months Sentenced")
    # plt.ylabel("Number of Cases")
    # plt.show()

if __name__ == "__main__":
    # path = "verdict_penalty.csv"
    # path = "pipline on test set.csv"
    # db = pd.read_csv(path, encoding='utf-8')
    # db = db.loc[db[SENTENCE] != "not found"]
    # time_db = extracted_time_db(db)
    # times = time_db[VOTED_TIME]
    # add_years(time_db)

    path = r"D:\PEAV\AssaultVerdictsParameterExtraction\times and years.csv"
    db = pd.read_csv(path,na_values = '', encoding='utf-8')
    db = db.loc[db["years"] != -1]
    db = db.loc[db[VOTED_TIME] != "-1.0"]
    db = db.loc[db[VOTED_TIME] != 4140.0]
    print("len db for years = ", len(db))
    values_by_col(db, VOTED_TIME)
    # values_by_col(db, "years")
    times = [float(x) for x in db[VOTED_TIME].to_list()]
    for t in times:
        if t > 500:
            print(t)

    print("median = ", np.median(times))
    print("average = ", np.average(times))
    print("min = ", np.min(times))
    print("max = ", np.max(times))
