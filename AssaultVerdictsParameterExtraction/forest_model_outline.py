# from AssaultVerdictsParameterExtraction.featureExtraction import extract
# import AssaultVerdictsParameterExtraction.validate
from sklearn.ensemble import RandomForestClassifier
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
CONTAINS_NUMBER = "does number appear"
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
            [[line[FILE_NAME], sentence, line[tag_name], prediction, probabilities, line[CONTAINS_NUMBER]]],
            columns=[FILE_NAME, SENTENCE, ORIGIN_TAG, PREDICTED_TAG, PROBA_VAL, CONTAINS_NUMBER])
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
    x_db = x_db.loc[:, x_db.columns != TAG_PROB]
    tag = db[tag_name]
    return x_db, tag

def cross_validation(db, tag_name, weight, test=True, soft_max = True):
    all_files = np.unique(db[FILE_NAME])
    np.random.seed(0)
    np.random.shuffle(all_files)
    file_num = len(all_files)
    cross_chunks = [all_files[x:x + int(file_num/9) ] for x in range(0, file_num, int(file_num/9))]
    print("chunk num = ", len(cross_chunks))
    recalls = []
    precisions = []
    f1_score = []
    ones = pd.DataFrame()
    after_max = pd.DataFrame(columns= [SENTENCE])
    for chunk in cross_chunks:
        temp_test = db.loc[db[FILE_NAME].isin(chunk)]
        temp_train = db.loc[~db[FILE_NAME].isin(chunk)]
        x_db, tag = train_and_tag_from_db(temp_train, tag_name)
        x_db = remove_strings(x_db)
        trained_model = train_func(x_db, tag, weight )

        if test:
            x, y = train_and_tag_from_db(temp_test, tag_name)
            x = remove_strings(x)
        else:
            x = x_db
            y = tag

        predicted_results, probabilities = predict_forest(x, trained_model)
        goal_labels = y.to_numpy()
        original_indices = y.index.to_numpy()
        rec, prec, f1, ones = check_prediction(predicted_results, goal_labels, weight, tag_name, original_indices, db,
                                               probabilities, ones, with_ones=True)
        if not soft_max:
            recalls.append(rec)
            precisions.append(prec)
            f1_score.append(f1)

    if soft_max:
        # ones = smaller_sentence_pool(predicted_results, tag_name, original_indices, db, probabilities)
        after_max, tn , fn = apply_argmax(ones, after_max, db)
        rec, prec, f1 = evaluate_prediction(after_max, weight, tn, fn)
        # rec, prec, f1 = check_prediction(after_max[PREDICTED_TAG], after_max[ORIGIN_TAG], weight, tag_name, original_indices, db,

        recalls.append(rec)
        precisions.append(prec)
        f1_score.append(f1)
    ones.to_csv("full_rf_prediction.csv", encoding="utf-8")
    print("for 10 cross_validation: recall = ", np.mean(recalls), " precision = ", np.mean(precisions),
          " and f1 = ", np.mean(f1_score))


def train_func(x_db, tag, weight):
    temp_db = x_db.loc[:, x_db.columns != SENTENCE]
    # x_train, x_test, y_train, y_test = train_test_split(temp_db, tag, shuffle=False, test_size=0.2, random_state=42)
    x_train = temp_db
    y_train = tag

    weights = (y_train.to_numpy() * weight) + 1

    clf = RandomForestClassifier(max_depth=2, random_state=0)  # kernel='linear')#, C=100)
    clf.fit(x_train, y_train, sample_weight=weights)

    return clf

def predict_forest(x, clf):
    # create predictions of which are the correct sentences
    x = x.loc[:, x.columns != SENTENCE]
    # x = remove_strings(x)
    predicted_results = clf.predict(x)
    probabilities = clf.predict_proba(x)
    return predicted_results, probabilities

def evaluate_prediction(after_max, weights, true_negative, false_negative):
    predicted_results = after_max[PREDICTED_TAG]
    true_positive = sum(after_max[ORIGIN_TAG])
    all_positive = len(after_max[ORIGIN_TAG])
    false_positive = all_positive - true_positive
    recall = true_positive / all_positive
    percision = (true_positive * weights) / (true_positive * weights + false_positive)
    f1 = calc_F1(true_positive * weights, true_negative, false_negative * weights, false_positive)
    print("sample size: ", all_positive)
    print("how many ones expected:", all_positive)
    print("how many ones predicted: ", sum(predicted_results))
    print("Precision: ", percision)
    print("Recall: ", recall)
    print("F1 score = ", f1)
    return  recall, percision, f1



def calc_F1(true_positive, true_negative, false_negative, false_positive):
    return true_positive / (true_positive + 0.5 * (false_positive + false_negative))

def apply_argmax(ones, after_max, db):
    true_negatives = 0
    false_negatives = 0
    files = np.unique(ones[FILE_NAME])

    for f in files:
        temp = ones.loc[ones[FILE_NAME] == f]
        if any(temp[CONTAINS_NUMBER]):
            temp = temp.loc[temp[CONTAINS_NUMBER] == True]
        max = temp[PROBA_VAL].argmax()

        s_line = temp.iloc[max]
        sentence = s_line[SENTENCE]
        tn = db.loc[((db[FILE_NAME] == f)& (db[TAG_COL] == 0) & db[SENTENCE] != sentence)]
        fn = db.loc[((db[FILE_NAME] == f)& (db[TAG_COL] == 1)&(db[SENTENCE]!= sentence))]
        true_negatives += len(tn)
        false_negatives += len(fn)
        # add_line(after_max, max, ORIGIN_TAG,1,temp,PROBA_VAL)
        after_max = after_max.append(dict(s_line), ignore_index= True)

        # print(max)
    after_max.to_csv("arg_max_output.csv", encoding="utf-8")
    return after_max, true_negatives, false_negatives



def check_prediction(predicted_results, goal_labels, weights, tag_name, X, df, probabilities, ones_db, with_ones = False):
    count_same = 0
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    ones = ones_db
    for i in range(len(predicted_results)):
        if predicted_results[i] == goal_labels[i]:
            count_same += 1
            if predicted_results[i] == 1:
                true_positive += 1
                if with_ones:
                    ones = add_line(ones, X[i], tag_name, predicted_results[i], df, probabilities[i][1])

            elif predicted_results[i] == 0:
                true_negative += 1

        elif goal_labels[i] == 1:
            false_negative += 1
        else:
            false_positive += 1
            if with_ones:
                ones = add_line(ones, X[i], tag_name, predicted_results[i], df, probabilities[i][1])

    recall = true_positive / sum(goal_labels)
    percision = (true_positive * weights) / (true_positive * weights + false_positive)
    f1 = 2*(percision*recall)/(percision + recall)
    print("sample size: ", len(goal_labels))
    print("how many ones expected:", sum(goal_labels))
    print("how many ones predicted: ", sum(predicted_results))
    print("Precision: ", percision)
    print("Recall: ",recall)
    print("F1 score = ", f1)

    if with_ones:
        ones.to_csv("ones_file.csv",encoding= 'utf-8')
        return recall, percision, f1, ones

    return recall, percision, f1



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
    path = r"D:\PEAV\AssaultVerdictsParameterExtraction\db_csv_files\feature_DB 28.07.csv"
    # path = r"D:\PEAV\AssaultVerdictsParameterExtraction\db_csv_files\DB of 27.6.csv"
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
    cross_validation(db_filtered, TAG_COL, 20, soft_max=False )
