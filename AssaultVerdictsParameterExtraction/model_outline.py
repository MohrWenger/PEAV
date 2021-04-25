# from AssaultVerdictsParameterExtraction.featureExtraction import extract
# import AssaultVerdictsParameterExtraction.validate
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
import pandas
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

SENTENCE = "sentence"
TAG_COL = "does sentence include punishment"
FILENAME = "filename"
#Recive a directory with .txt files
# path = 'verdicts'

## TRAIN ## - This is done only on our labled dataset

# call feature extraction return a db of all the sentences from all verdicts

# path = "C:\\Users\\oryiz\\PycharmProjects\\PEAV\\AssaultVerdictsParameterExtraction\\final_verdicts_dir\\"

path = "/Users/tomkalir/Projects/PEAV/AssaultVerdictsParameterExtraction/feature_DB - feature_DB (1).csv"
db = pandas.read_csv(path, header=0, na_values='')
s = 'does sentence include punishment'
db_filtered = db[(db[s] == 1) | (db[s] == 0)]
x_db = db_filtered.loc[:, db_filtered.columns != 'does sentence include punishment']
# x_db = x_db.loc[:, x_db.columns != 'sentence']
x_db = x_db.loc[:, x_db.columns != 'filename']

tag = db_filtered[s]


# vec = DictVectorizer()
# vectorized_db = vec.fit_transform(x_db).toarray()
# str_cols = x_db.columns[x_db.columns.str.contains('(?:filename|sentence)')]
# clfs = {c:LabelEncoder() for c in str_cols}
#
# for col, clf in clfs.items():
#     x_db[col] = clfs[col].fit_transform(x_db[col])
#
# str_cols = x_train.columns[x_train.columns.str.contains('(?:filename|sentence)')] #The coloumns of filename or sentence are held here
# clfs = {c:LabelEncoder() for c in str_cols} # A dictionary with a key and all the values for each coloumn
#
# for col, clf in clfs.items():
#     x_train[col] = clfs[col].fit_transform(x_train[col])


# str_cols = x_test.columns[x_test.columns.str.contains('(?:filename|sentence)')]
# clfs = {c:LabelEncoder() for c in str_cols}
#
# for col, clf in clfs.items():
#     x_test[col] = clfs[col].fit_transform(x_test[col])

# x_db = x_db.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
# tag = tag.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

# probably should use 10 fold cross validation from this point
# x_train, x_test, y_train, y_test = train_test_split(x_db, tag, random_state=0)

# clf = make_pipeline(StandardScaler(), SVC(gamma='auto')) -> I think this class several procedures sequentially.


def predict_func(x_db, test=True):
    x_db = x_db.replace(x_db[SENTENCE].tolist(), db_filtered.index.tolist())

    x_train, x_test, y_train, y_test = train_test_split(x_db, tag, test_size=0.2)

    weights = (y_train.to_numpy() * 7) + 1

    clf = SVC()
    clf.fit(x_train, y_train, sample_weight=weights)

    if test:
        x = x_test
        y = y_test
    else:
        x = x_train
        y = y_train

    # create predictions of which are the correct sentences
    predicted_results = clf.predict(x)

    y = y.to_numpy()
    x = x.to_numpy()

    fpr, tpr, thresholds = metrics.roc_curve(y, predicted_results, pos_label=1)


    count_ones = 0
    count_same = 0
    for i in range(len(predicted_results)):
        if predicted_results[i] == y[i]:
            count_same += 1
            if predicted_results[i] == 1:
                print(db_filtered.sentence[x[i][1]])
                count_ones += 1
    print("how many ones in train:", sum(y))
    print("how many ones predicted: ", sum(predicted_results))
    print("accuracy overall: ", count_same/len(predicted_results))
    print("accuracy of ones: ", count_ones/sum(y))
    print("AUC VALUE =", metrics.auc(fpr, tpr))
    metrics.plot_roc_curve(clf, x, y)
    plt.show()
# print(predicted_results)

##Test Train##
# evaluates predictions
predict_func(x_db)


## Post Train - single verdict ##