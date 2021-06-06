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
NUM_APPEAR = 'does time appear'
PROBA_VAL = "probability"
ORIGIN_TAG = "original tag"
PREDICTED_TAG = "our tag"
# path = 'verdicts'

## TRAIN ## - This is done only on our labled dataset

def weights_graphed(coef, db):
    imp = coef[0]
    imp, names = zip(*sorted(zip(imp, list(db.columns.values))))
    print(names)
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


def smaller_sentence_pool(predicted, tag_name, original_indices, db, probabilities):
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
            columns=[FILE_NAME, SENTENCE, ORIGIN_TAG, PREDICTED_TAG, PROBA_VAL])
        new_db = pd.concat([new_db, add])
    return new_db


### pre processing ###
def remove_strings(db):
    # db = db.replace(db[SENTENCE].tolist(), db.index.tolist())  # This is the line that replaces sentences
    # db.insert(0,"new col", db.index.tolist() )                                                              # with indx.
    db = db.set_index(pd.Index(np.arange(len(db[SENTENCE]))))  # with indx.
    le = LabelEncoder()
    file_name_dict = le.fit(db[FILE_NAME])
    db[FILE_NAME] = file_name_dict.fit_transform(db[FILE_NAME])

    return db


##Train##
def train_and_tag_from_db(db, tag_name):
    x_db = db.loc[:, db.columns != TAG_COL]
    x_db = x_db.loc[:, db.columns != TAG_PROB]
    tag = db[tag_name]
    return x_db, tag

def cross_validation(db, tag_name, weight, df, test=True):
    all_files = np.unique(db[FILE_NAME])
    file_num = len(all_files)
    cross_chunks = [all_files[x:x + int(file_num/5) ] for x in range(0, file_num, int(file_num/5))]

    for chunk in cross_chunks:
        temp_train = db.loc[db[FILE_NAME] not in chunk]
        temp_test = db.loc[db[FILE_NAME] in chunk]
        x_db, tag = train_and_tag_from_db(temp_train, tag_name)


        trained_model = train_func(x_db, tag, weight )
        if test:
            x, y = train_and_tag_from_db(temp_test)
        else:
            x = x_db
            y = tag

        check_prediction()

def train_func(x_db, tag, weight):
    x_db = remove_strings(x_db)

    np.random.seed(42)
    temp_db = x_db.loc[:, x_db.columns != SENTENCE]
    # x_train, x_test, y_train, y_test = train_test_split(temp_db, tag, shuffle=False, test_size=0.2, random_state=42)
    x_train = temp_db
    y_train = tag

    weights = (y_train.to_numpy() * weight) + 1

    clf = SVC(probability=True)  # kernel='linear')#, C=100)
    clf.fit(x_train, y_train, sample_weight=weights)

    return clf

def evaluate_prediction(x, y, clf, weight, tag_name, df):
    # create predictions of which are the correct sentences
    predicted_results = clf.predict(x)
    probabilities = clf.predict_proba(x)
    goal_labels = y.to_numpy()
    # X = x.to_numpy()
    original_indices = x.index.to_numpy()
    check_prediction(predicted_results, goal_labels, weight, tag_name, original_indices, df, probabilities)
    # weights_graphed(clf.coef_, x_train)

    ones = smaller_sentence_pool(predicted_results, tag_name, original_indices, df, probabilities)
    apply_argmax(ones)

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

    print("Amount of correct sentences in df = ", sum(ones['original tag']), ", amount predicted if taking last = ",
          count)
    print("Last sentence in file after SVM accuracy = ", count / sum(y))


def calc_F1(true_positive, true_negative, false_negative, false_positive):
    return true_positive / (true_positive + 0.5 * (false_positive + false_negative))

def apply_argmax(ones):
    files = np.unique(ones[FILE_NAME])
    after_max = pd.DataFrame(columns= [SENTENCE])
    for f in files:
        temp = ones.loc[ones[FILE_NAME] == f]
        #TODO - apply the actual argmax ... This is now by file names
        max = temp[PROBA_VAL].argmax()
        s_line = temp.iloc[max]
        # add_line(after_max, max, ORIGIN_TAG,1,temp,PROBA_VAL)
        after_max = after_max.append(dict(s_line), ignore_index= True)
        # print(max)
    after_max.to_csv("arg_max_output.csv", encoding="utf-8")



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
                # ones = add_line(ones, X[i], tag_name, predicted_results[i], df, propabilities[i][1])

            elif predicted_results[i] == 0:
                true_negative += 1

        elif goal_labels[i] == 1:
            false_negative += 1
        else:
            false_positive += 1
            # ones = add_line(ones, X[i], tag_name, predicted_results[i], df, propabilities[i][1])
    # TODO: FIGURE THIS LOOP OUT
    # ones = ones.sort_values(["file", "probability"], ascending=(True, False))
    # last_file = ""
    # new_ones = pd.DataFrame()
    # for line in ones:
    #     curr_file = line["file"]
    #     if curr_file != last_file:
    #         add = pd.DataFrame(
    #             [[line[FILE_NAME], line["sentence"], line[tag_name], line["probability"]]],
    #             columns=["file", "sentence", "original tag", "probability"])
    #         new_ones = pd.concat([new_ones, add])
    #     last_file = curr_file

    print("sample size: ", len(goal_labels))
    print("how many ones expected:", sum(goal_labels))
    print("how many ones predicted: ", sum(predicted_results))
    print("Precision: ", (true_positive * weights) / (true_positive * weights + false_positive))
    print("Recall: ", true_positive / sum(goal_labels))
    print("F1 score = ", calc_F1(true_positive * weights, true_negative, false_negative * weights, false_positive))


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

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 11, step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

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


def vizualize(db):
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

def remove_irrelevant_sentences(df):
    df = df.loc[df[NUM_APPEAR] == True]
    return df


if __name__ == "__main__":
    path = r"D:\PEAV\AssaultVerdictsParameterExtraction\DB of 30.5.csv"
    # path = "/Users/tomkalir/Projects/PEAV/AssaultVerdictsParameterExtraction/feature_DB - feature_DB (1).csv"
    # path = r"C:\Users\נועה וונגר\PycharmProjects\PEAV\AssaultVerdictsParameterExtraction\feature_DB - feature_DB (1).csv"
    db_initial = pd.read_csv(path, header=0, na_values='')
    db_filtered = db_initial
    # db_filtered = remove_irrelevant_sentences(db_initial)
    x_db = db_filtered.loc[:, db_filtered.columns != TAG_COL]
    x_db = x_db.loc[:, x_db.columns != TAG_PROB]
    tag = db_filtered[TAG_COL]
    # compute_PCA(db_filtered)
    # vizualize(db_filtered)
    for i in range(17, 30):  # 17 works good for actual
        print("with weights = ", i)
        train_func(x_db, tag, TAG_COL, i, db_filtered, test=True)
        print()
