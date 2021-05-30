# from AssaultVerdictsParameterExtraction.featureExtraction import extract
# import AssaultVerdictsParameterExtraction.validate
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
# import seaborn as sns
SENTENCE = "sentence"
FILE_NAME = "filename"
TAG_COL = "does sentence include punishment"
TAG_PROB = 'probation'
# path = 'verdicts'

## TRAIN ## - This is done only on our labled dataset

def weights_graphed(coef, db):
    imp = coef[0]
    imp, names = zip(*sorted(zip(imp, list(db.columns.values))))
    print(names)
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


def smaller_sentence_pool(predicted, tag_name,original_indices, db, probabilities):
    ones = pd.DataFrame()
    for i in range(len(predicted)):
        ones = add_line(ones, original_indices[i], tag_name, predicted[i], db, probabilities[i][1])
    ones.to_csv("svm_sentences.csv", encoding="utf-8")
    return ones


def add_line(new_db, line, tag_name, prediction, db, probabilities):
    sentence = db.sentence[line]
    line = db.iloc[line]
    if prediction == 1:
        add = pd.DataFrame(
                [[line[FILE_NAME], sentence, line[tag_name], prediction, probabilities]],
                columns=["file", "sentence", "original tag", "our tag", "probability"])
        new_db = pd.concat([new_db, add])
    return new_db


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
    x_train, x_test, y_train, y_test = train_test_split(temp_db, tag, shuffle=True, test_size=0.2, random_state=42)

    weights = (y_train.to_numpy() * weight) + 1

    clf = SVC(probability=True)  # kernel='linear')#, C=100)
    clf.fit(x_train, y_train, sample_weight=weights)

    if test:
        x = x_test
        y = y_test
    else:
        x = x_train
        y = y_train

    # create predictions of which are the correct sentences
    predicted_results = clf.predict(x)
    probabilities = clf.predict_proba(x)

    goal_labels = y.to_numpy()
    # X = x.to_numpy()
    original_indices = x.index.to_numpy()
    check_prediction(predicted_results, goal_labels, weight, tag_name,original_indices, df, probabilities)
    # weights_graphed(clf.coef_, x_train)

    ones = smaller_sentence_pool(predicted_results, tag_name,original_indices, df, probabilities)
    last_file = 0
    last_file_line = np.zeros((2, 4))
    count = 0
    for line in ones.iterrows():  # iterate through the subset of sentences the machine tagged as 1
        if line[1][0] != last_file:  # new file
            if last_file_line[1][2] == 1:  # check if last sentence was a manually tagged as one
                count += 1
        last_file_line = line
        last_file = line[1][0]
    ones.to_csv("svm_sentences.csv", encoding="utf-8")

    print("Amount of correct sentences in df = ", sum(ones['original tag']), ", amount predicted if taking last = ", count)
    print("Last sentence in file after SVM accuracy = ", count/sum(y))


def calc_F1 (true_positive, true_negative, false_negative, false_positive):
    return true_positive/(true_positive+0.5*(false_positive+false_negative))


def check_prediction(predicted_results, goal_labels, weights, tag_name, X, df, propabilities):
    count_same = 0
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    ones = pd.DataFrame()
    for i in range(len(predicted_results)):
        if predicted_results[i] == goal_labels[i]:
            count_same += 1
            if predicted_results[i] == 1:
                true_positive += 1
                ones = add_line(ones, X[i], tag_name, predicted_results[i], df, propabilities[i][1])

            elif predicted_results[i] == 0:
                true_negative += 1

        elif goal_labels[i] == 1:
                false_negative +=1
        else:
            false_positive += 1
            ones = add_line(ones, X[i], tag_name, predicted_results[i], df, propabilities[i][1])

    ones = ones.sort_values(["filename", "probability"], ascending=(True, False))
    last_file = ""
    new_ones = pd.DataFrame()
    for line in ones:
        curr_file = line["filename"]
        if
            add = pd.DataFrame(
                [[line[FILE_NAME], sentence, line[tag_name], prediction, probabilities]],
                columns=["file", "sentence", "original tag", "our tag", "probability"])
            new_db = pd.concat([new_db, add])

    print("sample size: ", len(goal_labels))
    print("how many ones expected:", sum(goal_labels))
    print("how many ones predicted: ", sum(predicted_results))
    print("Precision: ", (true_positive*weights)/(true_positive*weights + false_positive))
    print("Recall: ", true_positive/sum(goal_labels))
    print("F1 score = ", calc_F1(true_positive*weights, true_negative, false_negative*weights, false_positive))


#### visualization ####
def compute_PCA(db):
    from sklearn.preprocessing import MinMaxScaler
    del db["filename"]
    del db["sentence"]
    print(db.shape)
    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(db)
    pca = PCA().fit(data_rescaled)

    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (1804, 320)

    fig, ax = plt.subplots()
    xi = np.arange(1, 11, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 11, step=1)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

    ax.grid(axis='x')
    plt.show()
    #
    # db = remove_strings(db)
    # transformer = SparsePCA(n_components=2, random_state=0)
    # res = transformer.fit_transform(db)
    # df = pd.DataFrame()
    # df["y"] = db[TAG_COL]
    # df["comp-1"] = res[:, 0]
    # df["comp-2"] = res[:, 1]
    # sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
    #             palette=sns.color_palette("hls", 2),
    #             data=df).set(title="Iris data SparsePCA projection")
    # plt.show()

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

def remove_x_first(db):
    files = np.unique(db[FILE_NAME])
    for f in files:
        temp_df = []

if __name__ == "__main__":
    path = r"D:\PEAV\AssaultVerdictsParameterExtraction\DB of 30.5.csv"
    # path = "/Users/tomkalir/Projects/PEAV/AssaultVerdictsParameterExtraction/feature_DB - feature_DB (1).csv"
    # path = r"C:\Users\נועה וונגר\PycharmProjects\PEAV\AssaultVerdictsParameterExtraction\feature_DB - feature_DB (1).csv"
    db_initial = pd.read_csv(path, header=0, na_values='')
    # s = 'does sentence include punishment'
    # db_filtered = db[(db[s] == 1) | (db[s] == 0)]
    db_filtered = db_initial
    x_db = db_filtered.loc[:, db_filtered.columns != TAG_COL]
    x_db = x_db.loc[:, x_db.columns != TAG_PROB]
    tag = db_filtered[TAG_COL]
    # compute_PCA(db_filtered)
    # vizualize(db_filtered)
    for i in range(17,30): #17 works good for actual
        print("with weights = ", i)
        train_and_predict_func(x_db,tag, TAG_COL,i, db_filtered,test=True )
        print()