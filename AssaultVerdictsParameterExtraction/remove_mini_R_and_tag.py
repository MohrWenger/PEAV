import json
from tqdm import tqdm
import os
import pandas as pd
from AssaultVerdictsParameterExtraction import penalty_extraction as pe
from AssaultVerdictsParameterExtraction import featureExtraction as fe
import math
SUMMARY_COL = 'תקציר'
FILE_NAME_COL = 'קובץ 4'
MINI_RATIO = 'מיני-רציו'
FILE_NAME = "filename"
TAG_COL = "does sentence include punishment"
SENTENCE = "sentence"

#Open the Igud2021.xls as pd
path = "db_csv_files/Igud2021.xlsx - igud12020.csv"
Igud_db = pd.read_csv(path, encoding='utf-8')

#File name from test casename
directory = "final_verdicts_dir"
db = pd.DataFrame()
with open('test_case_filenames.txt') as json_file:
    relevant_cases = json.load(json_file)  # Cases of the validation file
    for i, filename in tqdm(enumerate(os.listdir(directory))):  # when iterating through all files in folder
        if filename.endswith(".txt") and filename in relevant_cases:
            name = filename.split(".")[0]

            # look for the filename without .txt in coloumn "קובץ 4" and extract the line where it appears
            temp = Igud_db.loc[Igud_db[FILE_NAME_COL].astype(str).str.contains(name)]
            # (If there is more than one check if they are all Gzar)
            # print("name = ", name, "for row = ", temp[FILE_NAME_COL])
            summary = str(temp[SUMMARY_COL])
            if MINI_RATIO in summary: # average len of mini ratio is 82 - 83 chars.
                # read text
                print(len(summary), summary)
            file_name = os.path.join(directory, filename)
            text = open(file_name, "r", encoding="utf-8").read()
            mini_ratio_index = text.find(MINI_RATIO)

            if text.find(MINI_RATIO) != -1:
                #remove all text up to 90 chars after mini ratio
                text = text[mini_ratio_index + 90:]

            to_add = fe.add_sentences_with_features(filename, text, db)
            db = pd.concat([db, to_add])

            # call the relevant function from penalty extraction

#### After creating the DB tag it with the previous database
db[TAG_COL] = 0
path_tagged = r"D:\PEAV\AssaultVerdictsParameterExtraction\db_csv_files\feature_DB 28.07.csv"
tagged_db = pd.read_csv(path_tagged)
for index, row in db.iterrows():
    temp = tagged_db[(tagged_db[FILE_NAME] == str(row[FILE_NAME])) & (tagged_db[SENTENCE] == str(row[SENTENCE]))]
    if not temp.empty:
        db.at[index, TAG_COL] = int(temp[TAG_COL])
    # if (tagged_db[FILE_NAME] == row[FILE_NAME]) and (tagged_db[SENTENCE] == row[SENTENCE]):
db.to_csv("without_mini_ratio.csv")

