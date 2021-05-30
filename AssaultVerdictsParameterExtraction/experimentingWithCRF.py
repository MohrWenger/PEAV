import pandas as pd
import numpy as np
#Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set(font_scale=1)
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
#Modeling

from sklearn.model_selection import cross_val_predict, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import classification_report, make_scorer
import scipy.stats
import eli5

PATH = "D:\PEAV\AssaultVerdictsParameterExtraction\GMB_dataset.txt"
data = pd.read_csv(PATH, sep="\t", header=None, encoding="latin1")
data.head()
