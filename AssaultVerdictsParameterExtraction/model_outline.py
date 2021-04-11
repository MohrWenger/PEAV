from AssaultVerdictsParameterExtraction.featureExtraction import extract
from AssaultVerdictsParameterExtraction.featureExtraction import validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Recive a directory with .txt files
path = 'verdicts'

## TRAIN ## - This is done only on our labled dataset

# call feature extraction return a db of all the sentences from all verdicts
path = "C:\\Users\\oryiz\\PycharmProjects\\PEAV\\AssaultVerdictsParameterExtraction\\final_verdicts_dir\\"
db = extract(path,2)
x_db = db.loc[:, db.columns != 'label']
labels = db.loc['label']

# probably should use 10 fold cross validation from this point
x_train, x_test, y_train, y_test = train_test_split (x_db, labels, random_state = 0)
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto')) -> I think this class several procedures sequentaly.
clf = SVC()
clf.fit(x_train, y_train)

# create predictions of which are the correct sentences
predicted_results = clf.predict(x_test)

##Test Train##
# evaluates predictions
''' Toms function for evaluation'''

## Post Train - single verdict ##