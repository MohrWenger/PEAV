# from AssaultVerdictsParameterExtraction.featureExtraction import extract
# import AssaultVerdictsParameterExtraction.validate
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import SparsePCA
import seaborn as sns
SENTENCE = "sentence"
FILE_NAME = "filename"
TAG_COL = "does sentence include punishment"
TAG_PROB = 'probation'
# path = 'verdicts'

## TRAIN ## - This is done only on our labled dataset

# call feature extraction return a db of all the sentences from all verdicts

# x_db = x_db.loc[:, x_db.columns != 'sentence']
# x_db = x_db.loc[:, x_db.columns != 'filename']


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




# for col, clf in clfs.items():
#     x_test[col] = clfs[col].fit_transform(x_test[col])

# probably should use 10 fold cross validation from this point
# x_train, x_test, y_train, y_test = train_test_split(x_db, tag, random_state=0)

def weights_graphed(coef, db):
    imp = coef[0]
    imp, names = zip(*sorted(zip(imp, list(db.columns.values))))
    print(names)
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


def smaller_sentence_pool(predicted, tag_name,original_indices, db):
    ones = pd.DataFrame()
    for i in range(len(predicted)):
        # sentence = db.sentence[x[i][1]]
        sentence = db.sentence[original_indices[i]]
        # print(sentence)
        # line = db.loc[db[SENTENCE] == sentence]
        line = db.iloc[original_indices[i]]
        if predicted[i] == 1:
            add = pd.DataFrame(
                    [[line[FILE_NAME], sentence, line[tag_name], predicted[i]]],
                    columns=["file", "sentence", "original tag", "our tag"])
            ones = pd.concat([ones, add])
    ones.to_csv("svm_sentences.csv", encoding="utf-8")
    return ones

### pre processing ###
def remove_strings (db):
    # db = db.replace(db[SENTENCE].tolist(), db.index.tolist())  # This is the line that replaces sentences
    # db.insert(0,"new col", db.index.tolist() )                                                              # with indx.
    db = db.set_index(pd.Index(np.arange(len(db[SENTENCE]))))                                                              # with indx.
    le = LabelEncoder()
    file_name_dict = le.fit(db[FILE_NAME])
    db[FILE_NAME] = file_name_dict.fit_transform(db[FILE_NAME])

    return db

##Train##

def train_and_predict_func(x_db, tag, tag_name, weight, df,test=True):
    x_db = remove_strings(x_db)

    np.random.seed(42)
    temp_db = x_db.loc[:, x_db.columns != SENTENCE]
    x_train, x_test, y_train, y_test = train_test_split(temp_db, tag, shuffle= True,test_size=0.2, random_state=42)

    weights = (y_train.to_numpy() * weight) + 1

    clf = SVC()#kernel='linear')#, C=100)
    clf.fit(x_train, y_train, sample_weight=weights)

    if test:
        x = x_test
        y = y_test
    else:
        x = x_train
        y = y_train

    # create predictions of which are the correct sentences
    predicted_results = clf.predict(x)

    goal_labels = y.to_numpy()
    # X = x.to_numpy()
    original_indices = x.index.to_numpy()
    check_prediction(predicted_results, goal_labels, tag_name,original_indices, df)
    # weights_graphed(clf.coef_, x_train)

    ones = smaller_sentence_pool(predicted_results, tag_name,original_indices, df)
    # last_file = 0
    # count = 0
    # for line in ones:
    #     if line[0] != last_file:
    #         if line[2] == 1:
    #             count += 1
    #     last_file = line[0]
    # print("Last sentence in file after SVM accuracy = ", count/sum(y))


def check_prediction(predicted_results, goal_labels, tag_name ,X, df):
    count_ones = 0
    count_same = 0
    for i in range(len(predicted_results)):
        # if goal_labels[i] == 1:
        #     print("break point")
        if predicted_results[i] == goal_labels[i]:
            count_same += 1
            if predicted_results[i] == 1:
                # print(X[i])
                # print(df.sentence[X[i]])
                print(df[tag_name][X[i]], "and goals = ", goal_labels[i])
                count_ones += 1
        elif goal_labels[i] == 1:
            print(df.sentence[X[i]])
    print("sample size: ", len(goal_labels))
    print("how many ones expected:", sum(goal_labels))
    print("how many ones predicted: ", sum(predicted_results))
    print("Precision: ", count_same/len(predicted_results))
    print("Recall: ", count_ones/sum(goal_labels))

    # fpr, tpr, thresholds = metrics.roc_curve(goal_lables, predicted_results, pos_label=1)
    # print("AUC VALUE =", metrics.auc(fpr, tpr))

    # metrics.plot_roc_curve(clf, x, y)
    # plt.show()
    # one_hot_encoding(predicted_results, x, y, db_filtered)


#### visualization ####
def compute_PCA (db):
    db = remove_strings(db)
    transformer = SparsePCA(n_components=2, random_state=0)
    res = transformer.fit_transform(db)
    df = pd.DataFrame()
    df["y"] = db[TAG_COL]
    df["comp-1"] = res[:, 0]
    df["comp-2"] = res[:, 1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="Iris data SparsePCA projection")
    plt.show()

def vizualize (db):
    db = remove_strings(db)
    # correct = db.loc[db[TAG_COL] == 1]
    # false_sentences = db.loc[db[TAG_COL] == 0]
    tsne_res = TSNE(n_components=2).fit_transform(db)
    # false_embedded = tsne.fit_transform(false_sentences)
    # correct_embedded = tsne.fit_transform(correct)
    # print(X_embedded.shape)
    # plt.scatter(false_embedded[:,0], false_embedded[:,1])
    # plt.scatter(correct_embedded[:,0], correct_embedded[:,1])

    # df_subset = pandas.DataFrame(tsne_res, columns =['tsne-2d-one', 'tsne-2d-two'])
    db['tsne-2d-one'] = tsne_res[:, 0]
    db['tsne-2d-two'] = tsne_res[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=TAG_COL,
        palette=sns.color_palette("hls", 2),
        data=db,
        legend="full"
    )
    plt.savefig("tsne_try1.png")
    plt.show()
## Post Train - single verdict ##


if __name__ == "__main__":
    path = r"D:\PEAV\AssaultVerdictsParameterExtraction\DB of 9.5 - feature_DB.csv"
    # path = "/Users/tomkalir/Projects/PEAV/AssaultVerdictsParameterExtraction/feature_DB - feature_DB (1).csv"
    # path = r"C:\Users\נועה וונגר\PycharmProjects\PEAV\AssaultVerdictsParameterExtraction\feature_DB - feature_DB (1).csv"
    db_initial = pd.read_csv(path, header=0, na_values='')
    # s = 'does sentence include punishment'
    # db_filtered = db[(db[s] == 1) | (db[s] == 0)]
    db_filtered = db_initial
    x_db = db_filtered.loc[:, db_filtered.columns != TAG_COL]
    x_db = x_db.loc[:, x_db.columns != TAG_PROB]
    tag = db_filtered[TAG_PROB]
    # compute_PCA(db_filtered)
    # vizualize(db_filtered)
    for i in range(17,30): #17 works good for actual
        print("with weights = ", i)
        train_and_predict_func(x_db,tag, TAG_PROB,i, db_filtered )
        print()