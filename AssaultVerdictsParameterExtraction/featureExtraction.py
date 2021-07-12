import re
import os
import pandas as pd
from AssaultVerdictsParameterExtraction import penalty_extraction as pe
import json
from tqdm import tqdm

COL_HEB_NAMES = []
COL_CLAUSES = []

# Order of copying: list of bad words, list of bad signs list of good words list of moderate words
HEB_WORDS_TO_EXTRACT = ['עו*תרה*(ים)*(ות)*','ה*תובעת*','ביקשה*','ה*תביעה',"ה*סנגור","בי*קשה*","ערעור",
                        'מבחן','צבאי','בי*טחון','קבע','דורשת*','בימים',
                        'בין','מתחם','יפחת','יעלה','נגזר','נדון','ה*צדדים',"\"","/",r"\\",":"
                        ,'גוזרת*(ים)*(ות)*'
                        ,"ריצוי בפועל","מה[ןם]","היתרה","מתו*כ[ןם]"
                        #' [1-9]+ *חודשי מאסר',' [1-9]+ *שנות מאסר',
                        ,'[נמ]טילה*(ים)*(ות)*',' ד[(נה)(ן)(נים)(נות)]','משיתה*(ים)*(ות)*','מחליטה*(ים)*(ות)*','לגזור','להטיל'
                        #'יי*מצא מתאים', 'מאסר', 'עונשין', 'שירות', 'תנאי', 'יעבור',
                        ,'עבודות', 'צו', 'קהיל[הת]י?', 'ציבור'
                        ,'תועלת', 'נאשם', 'קנס','שי*רות המבחן','אלא אם','מו*תו*נ[(ה)(ים)]',' ב*סך','כס[ףפ]' #,'על תנאי','שנ[(ות)(ים)] מאסר','345','351','346','347','348','349','350'
                        ,'3 שנים','אנ[יו] '] + pe.TIME_UNITS_ARR

for word in HEB_WORDS_TO_EXTRACT:
    COL_HEB_NAMES.extend(["first " + word, "first " + word + " ratio", "last " + word, "last " + word + " ratio", "did " +word+ " appear",
                          word + " count"])

for col in pe.CLAUSES:
    COL_CLAUSES.append(["first " + col, "first " + col + " ratio", "last " + col, "last " + col + " ratio",
                        col + " count"])

SYNTAX_NAMES = ["NN", "VB", "PREPOSITION", "DEF", "NNP", "NNT", "BN", "POS", "S_PRN", "CD", "RB", "PRP",
                "IN", "AT", "DTT", "yyCM", "CONJ", "JJ", "COP", "REL", "NCD", "yyDOT", "gen=M", "gen=F"]


def extract_important_words(sentence, words):
    list_of_indices = []
    for word in words:
        indices = [index.start() for index in re.finditer(word, sentence)]
        length = len(indices)
        list_of_indices.append([-1 if length == 0 else indices[0],
                                -1 if length == 0 else indices[0]/len(sentence),
                                -1 if length == 0 else indices[-1],
                                -1 if length == 0 else indices[-1]/len(sentence),
                                -1 if length == 0 else 1,
                                length])
        # else:
        # list_of_indices.append([ #indices[0],
        #                          indices[0]/len(sentence),
        #                          # 0,
        #                          0,
        #                          1,
        #                          length
        #                         ])
    return list_of_indices

def does_contain_number(sentence):
    """
    This function returns if a sentence contains a number
    :param sentence:
    :return:
    """
    new_sentence = pe.numberExchange(sentence)
    smaller_then_50 = 0
    bigger_then_300 = 0
    contains = False
    numbers = re.findall('[1-9]+',new_sentence)
    for num in numbers:
        contains = True
        if float(num) < 50:
            smaller_then_50 += 1
        elif float(num) > 1000:
            bigger_then_300 += 1


    return contains, smaller_then_50, bigger_then_300


def add_syntax_features(featureDB):
    txt = open('old_attempts/output.conll', 'r').read()
    for name in SYNTAX_NAMES:
        i_next = 0
        column = []
        for i in range(len(featureDB)):
            i_prev = i_next + 1
            i_next = txt[i_prev:].find("\n1\t") + i_prev
            column.append(len(re.findall(name, txt[i_prev:i_next])) / 2)
        featureDB[name] = column
    return featureDB



def operating_func(filename, featureDB):
    text = open(path + filename, "r", encoding="utf-8").read()
    sentence_allfile_count = len(text.split("."))
    sentences, len_sentences, sent_num = pe.extracting_penalty_sentences(text, True)

    for i in range(len(sentences)):
        important_words_list = extract_important_words(sentences[i], HEB_WORDS_TO_EXTRACT)
        # clauses_list = extract_important_words(sentences[i], pe.CLAUSES)
        bool_number,small_nums, big_nums = does_contain_number(sentences[i])

        # here I add values to DB
        sentence_line = pd.DataFrame(
            [[filename, sentences[i], len_sentences[i], sentence_allfile_count, sent_num[i], bool_number, small_nums, big_nums,
              sent_num[i]/sentence_allfile_count] +
             [j for i in important_words_list for j in i] + [0] * len(SYNTAX_NAMES)],
            columns=["filename", "sentence", "length sentence", "total sentences", "sentence num",
                     "does number appear","smaller than 50","bigger than 100","ratio in file"] + COL_HEB_NAMES + SYNTAX_NAMES)
        featureDB = pd.concat([featureDB, sentence_line])
    # return add_syntax_features(featureDB)
    return featureDB

def find_best_params(featureDB):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    print(featureDB.shape)
    print(np.mean(featureDB),np.std(featureDB))
    pca_breast = PCA(n_components=2)
    x = StandardScaler().fit_transform(featureDB)
    principalComponents_breast = pca_breast.fit_transform(x)



def extract(directory, running_opt):
    featureDB = pd.DataFrame()

    if running_opt == 0:
        with open('test_case_filenames.txt') as json_file:
            relevant_cases = json.load(json_file)  # Cases of the validation file
            for i, filename in tqdm(enumerate(os.listdir(directory))):  # when iterating through all files in folder
                if filename.endswith(".txt") and filename in relevant_cases:
                    if filename == "m02001206.txt":
                        print("break point")
                    featureDB = operating_func(filename, featureDB)
                    # featureDB = pd.concat([featureDB, sentence_line])

    elif running_opt == 2:
        counter = 0
        for filename in os.listdir(directory):
            if counter < 150:
                featureDB = operating_func(filename, featureDB)
                # featureDB = pd.concat([featureDB, sentence_line])
                counter += 1

    featureDB.to_csv("feature_DB.csv", encoding="utf-8")
    print("number of cases = ", len(featureDB["filename"]))
    return featureDB


if __name__ == "__main__":
    # path = "/Users/tomkalir/Projects/PEAV/AssaultVerdictsParameterExtraction/final_verdicts_dir/"
    path = r"final_verdicts_dir/"
    extract(path, 0)
    print(len(HEB_WORDS_TO_EXTRACT))




