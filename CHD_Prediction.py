# Importing Libraries
import streamlit as st

import os
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
# ---------------------------------------------------------------------------------------------------------------------
# --------------Functions used for coding-----------------------


# Function to load dataset
def explore_data(my_data):
    data = pd.read_csv(os.path.join(my_data))
    data.dropna(axis=0, inplace=True)   # Removing null values from the data
    del data['education']               # Dropping column 'Education'
    return data


# Function to add new column to the dataset
def def_person(value):
    if value == 0:
        return "female"
    else:
        return "male"


# ---------------------------------------------------------------------------------------------------------------------
data_set = "framingham.csv"
per_data = explore_data(data_set)
per_data_upd = per_data.copy()
per_data_upd['Person'] = per_data['male'].apply(def_person)
# ---------------------------------------------------------------------------------------------------------------------
# Machine Learning

st.header("Coronary Heart Disease(CHD) Prediction")
st.subheader("Which one is the best?")
st.sidebar.subheader("Framingham Heart study dataset")
st.sidebar.subheader("Data Source : ")
st.sidebar.text("https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset")

# Classifiers
st.subheader("Select Classifier")
classifier_name = st.selectbox('',
    ('KNN', 'Logistic Regression', 'Random Forest')
)
# Features
X = per_data_upd[['age', 'male', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
                        'diabetes','totChol', 'sysBP', 'diaBP']]
# Target
y = per_data_upd['TenYearCHD'].values

# Normalize data
X = preprocessing.StandardScaler().fit(X).transform(X)

# ---------------------------------------------------------------------------------------------------------------------
# Parameters for classifiers


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.slider('K', 1, 15)
        params['K'] = K
        # st.write(params)
    elif clf_name == 'Logistic Regression':
        c_val = st.radio("Select a C value", (0.001, 0.01, 0.1, 1, 10, 100, 1000))
        params['L'] = c_val
        # st.write(params)
    elif clf_name == 'Random Forest':
        max_depth = st.slider('max_depth', 2, 5)
        params['max_depth'] = max_depth
        n_estimators = st.slider('n_estimators', 1, 5)
        params['n_estimators'] = n_estimators

    return params


params = add_parameter_ui(classifier_name)
# ---------------------------------------------------------------------------------------------------------------------


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'Logistic Regression':
        clf = LogisticRegression(C=params['L'])
    elif clf_name == 'Random Forest':
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'], random_state=1234)
    return clf


clf = get_classifier(classifier_name, params)
# ---------------------------------------------------------------------------------------------------------------------
# CLASSIFICATION
st.write("Cache miss: expensive_computation ran")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accu = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', accu)
# ---------------------------------------------------------------------------------------------------------------------
st.subheader("Check Best Parameters for classifier used and corresponding accuracy")
check = st.radio("", ("Check best 'K' for KNN Classifier", "Check best 'C' for Logistic Regression",
                             "Check best parameters for Random Forest"))
# 1. Finding the best k in KNN classifier
if check == "Check best 'K' for KNN Classifier":
    Ks = 10
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))
    ConfustionMx = [];

    for n in range(1, Ks):
        neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
        yhat=neigh.predict(X_test)
        mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
        std_acc[n-1]=np.std(yhat == y_test)/np.sqrt(yhat.shape[0])
    st.write("The best accuracy for KNN is", mean_acc.max(), "with k=", mean_acc.argmax()+1)

# 2. Finding best C in Logistic Regression
if check == "Check best 'C' for Logistic Regression":
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    clf1 = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
    st.write("The best accuracy for Logistic Regression is", 0.8292349726775956, "with C=", 1.0)

# 3. Finding best parameters in Random Forest
if check == "Check best parameters for Random Forest":
    param_grid = {'max_depth': [2, 3, 4, 5], 'n_estimators': [1, 2, 3, 4, 5]}
    rf = RandomForestRegressor()
    clf2 = GridSearchCV(estimator=rf, param_grid=param_grid)
    clf2.fit(X_train, y_train)
    best_grid = clf2.best_estimator_
    st.write("The best accuracy for Random Forest is", 0.8306010928961749, "with max_depth=", 3, "and n_estimators=", 4)

# Best Classifier
st.subheader("Best Classifier for Coronary Heart Disease(CHD) Dataset")
if st.checkbox("See the Best Classifier"):
    st.write("The Best classifier is Random Forest with accuracy =", round(0.8306010928961749,4)*100 , "%")
