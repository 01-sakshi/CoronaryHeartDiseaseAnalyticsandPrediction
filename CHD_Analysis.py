# Importing Libraries
import streamlit as st

import os
import pandas as pd
import numpy as np


from matplotlib import pyplot as plt

import seaborn as sns
import scipy.stats as stats

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

# ---------------------------------------------------------------------------------------------------------------------


# Function to add new column to the dataset
def def_person(value):
    if value == 0:
        return "female"
    else:
        return "male"

# ---------------------------------------------------------------------------------------------------------------------


# Function for styling  of bar-plots
def bar_plot_style():
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    st.pyplot()


# ---------------------------------------------------------------------------------------------------------------------
data_set = "framingham.csv"
per_data = explore_data(data_set)
per_data_upd = per_data.copy()
per_data_upd['Person'] = per_data['male'].apply(def_person)
# ---------------------------------------------------------------------------------------------------------------------


# Function for App Description
def app_descrip():
    st.title("""Coronary Heart Disease(CHD) Analytics""")
    st.sidebar.subheader("Framingham Heart study dataset")
    st.sidebar.subheader("Data Source : ")
    st.sidebar.text("https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset")
    st.sidebar.text("""
    Coronary heart disease (CHD), or 
    coronary artery disease,develops 
    when the coronary arteries become too
    narrow.The coronary arteries are the
    blood vessels that supply oxygen and 
    blood to the heart.CHD tends to 
    develop when cholesterol builds up on
    the artery walls,creating plaques. 
    CHD can sometimes lead to heart
    attack. It is the “most common type
    of heart disease in the United 
    States,” where it accounts for more 
    than 370,000 deaths every year.
    The early prognosis of cardiovascular
    diseases can aid in making decisions
    on lifestyle changes in high risk 
    patients and in turn reduce the 
    complications.This research intends
    to pinpoint the most relevant/risk
    factors of heart disease as well
as predict the overall risk.""")


# ---------------------------------------------------------------------------------------------------------------------


# Function for Data Understanding
def data_understanding():
    st.header("Data Description")
    # Showing dataset
    if st.checkbox("Preview Dataset"):
        if st.button("Head"):
            st.subheader("Top 5 rows of the data")
            st.write(per_data.head())
        if st.button("Tail"):
            st.subheader("Last 5 rows of the data")
            st.write(per_data.tail())

    # Show entire dataset
    if st.checkbox("Show entire dataset"):
        st.write(per_data)

    # Show Column Names
    if st.checkbox("Show Column names"):
        st.write(per_data.columns)

    # Show dataframe dimensions
    st.subheader("Dataset dimensions")
    per_data_dim = st.radio("Which dimension you want to see?", ("Rows", "Columns", "All"))
    if per_data_dim == 'Rows':
        st.text("Showing Rows dimension")
        st.write(per_data.shape[0])
    if per_data_dim == 'Columns':
        st.text("Showing Columns dimension")
        st.write(per_data.shape[1])
    if per_data_dim == 'All':
        st.text("Showing shape of dataset")
        st.write(per_data.shape)

    # Show summary of dataset
    if st.checkbox("Show summary of dataset"):
        st.write(per_data.describe())

    # Selection of Columns
        st.subheader("Show Column Details")
        per_data_cols = st.selectbox('Select Column', ('male', 'age', 'currentSmoker',
          'cigsPerDay', 'BPMeds', "prevalentStroke",
         "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "TenYearCHD"))
        if per_data_cols == 'male':
            st.write(per_data['male'])
        elif per_data_cols == 'age':
            st.write(per_data['age'])
        elif per_data_cols == 'currentSmoker':
            st.write(per_data['currentSmoker'])
        elif per_data_cols == 'cigsPerDay':
            st.write(per_data['cigsPerDay'])
        elif per_data_cols == 'BPMeds':
            st.write(per_data['BPMeds'])
        elif per_data_cols == 'prevalentStroke':
            st.write(per_data['prevalentStroke'])
        elif per_data_cols == 'prevalentHyp':
            st.write(per_data['prevalentHyp'])
        elif per_data_cols == 'diabetes':
            st.write(per_data['diabetes'])
        elif per_data_cols == 'totChol':
            st.write(per_data['totChol'])
        elif per_data_cols == 'sysBP':
            st.write(per_data['sysBP'])
        elif per_data_cols == 'diaBP':
            st.write(per_data['diaBP'])
        elif per_data_cols == 'BMI':
            st.write(per_data['BMI'])
        elif per_data_cols == 'heartRate':
            st.write(per_data['heartRate'])
        elif per_data_cols == 'glucose':
            st.write(per_data['glucose'])
        elif per_data_cols == 'TenYearCHD':
            st.write(per_data['TenYearCHD'])
        else:
            st.write("Select A Column")


# ---------------------------------------------------------------------------------------------------------------------
# Functions for Exploratory Data Analysis


# Gender distribution in the data
# @st.cache(suppress_st_warning=True)
def gender_dist():
    st.header("Data Analysis")
    st.subheader("Gender distribution in the data")
    plt.subplots(figsize=(6, 4))
    g = sns.catplot(y=None, x="Person", palette="Blues", data=per_data_upd, kind='count')
    g.set_axis_labels("Gender", "No. of people")
    sns.set_style("whitegrid", {'axes.grid': False})
    st.pyplot()


# Age data distribution in the dataset
# @st.cache(suppress_st_warning=True)
def age_dist():
    st.subheader("Age data distribution in the dataset")
    plt.figure(figsize=(5, 5))
    age = per_data['age']
    plt.hist(age, rwidth=0.9, color='steelblue', bins=[30, 40, 50, 60, 70])
    plt.title('Distribution of Age data', alpha=0.8)
    plt.xlabel("Age", fontsize=14, alpha=0.8)
    plt.ylabel("No. of people", fontsize=14, alpha=0.8)
    st.pyplot()


# 10 year risk of Coronary Heart Disease(CHD) data distribution
# @st.cache(suppress_st_warning=True)
def chd_dist():
    st.subheader("Coronary Heart Disease(CHD) data distribution in the dataset")

    g = sns.catplot(y=None, x="TenYearCHD", palette="Blues", data=per_data, kind='count')
    g.set_xticklabels(["Without CHD", "With CHD"])
    g.set_axis_labels("Ten Year CHD", "No. of people")
    st.pyplot()


# Gender vs Cigrettes per day(cigsPerDay)
# @st.cache(suppress_st_warning=True)
def gen_cig():
    st.subheader("Gender v/s Cigrettes per day")

    g = sns.catplot(x="Person", y="cigsPerDay", palette="GnBu_d", data=per_data_upd, kind='box')
    g.set_axis_labels("Gender", "Cigarettes per Day")
    st.pyplot()


# Heart Rate and Cholestrol
# @st.cache(suppress_st_warning=True)
def heart_chol():
    st.subheader("Heart Rate and Cholestrol")

    plt.figure()
    X = per_data['heartRate']
    Y = per_data['totChol']
    _ = plt.hist2d(X, Y)
    plt.xlabel("Heart Rate", alpha=0.8)
    plt.ylabel("Total Cholestrol", alpha=0.8)
    plt.title("Heart Rate and Cholestrol 2-D plot", alpha=0.8)
    plt.colorbar()
    plt.show()
    st.pyplot()


# Cigarettes per day and Cholestrol
# @st.cache(suppress_st_warning=True)
def cigs_chol():
    st.subheader("Cigarettes per day and Cholestrol")

    plt.figure()
    X = per_data['cigsPerDay']
    Y = per_data['totChol']
    _ = plt.hist2d(X, Y)
    plt.xlabel("Cigarettes per day", alpha=0.8)
    plt.ylabel("Total Cholestrol", alpha=0.8)
    plt.title("Cigarettes per day and Cholestrol 2-D plot", alpha=0.8)
    plt.colorbar()
    plt.show()
    st.pyplot()


# Distribution of data in all the columns
# @st.cache(suppress_st_warning=True)
def dist_all_data():
    st.subheader("Distribution of data in all columns")

    fig = plt.figure(figsize=(25, 25))

    for i, feature in enumerate(per_data.columns):
        ax = fig.add_subplot(5, 3, i+1)
        s = per_data[feature].hist(bins=20, ax=ax, facecolor='steelblue')
        ax.set_title( feature + " Distribution" , color = 'red' )
    st.pyplot()


# Plotting mean and standard deviation of the data
# @st.cache(suppress_st_warning=True)
def mean_sd():
    st.subheader("Plotted mean and standard deviation of each column")

    mean_data = per_data.mean()
    std_data = per_data.std()
    fig = plt.figure(figsize = (18,12))
    ax = fig.add_subplot(111)

    mean_data.plot(legend=False , kind="bar" , rot=45 , color="royalblue",
                                fontsize=16, yerr=std_data ,alpha=0.8)
    ax.set_title("Plotted mean and standard deviation of each column", fontsize=18, alpha=0.8)
    ax.set_xlabel("Features", fontsize=18 , alpha=0.8)
    ax.set_ylabel("Values", fontsize=18 , alpha=0.8)
    st.pyplot()


# Normal Distribution plotted for each column.
# @st.cache(suppress_st_warning=True)
def normal_dist():
    st.subheader("Normal Distribution plotted for each column")

    fig = plt.figure(figsize=(20, 20))

    for i, feature in enumerate( per_data.columns ):
        x = per_data[feature].tolist()
        x.sort()
        ax = fig.add_subplot(5, 3, i+1)
        pdf = stats.norm.pdf(x,  np.mean(x),  np.std(x))
        plt.plot(x, pdf)
        ax.set_title(feature + " Distribution", color = 'red')
    st.pyplot()


# Most relevant risk factors for heart disease

# Age v/s Ten Year CHD Risk  Distribution
# @st.cache(suppress_st_warning=True)
def age_chd():
    st.subheader("Age v/s Ten Year CHD Risk  Distribution")

    fig = plt.figure( figsize = (7,7) )
    x = sorted(per_data['age'].unique())     # 39 unique values
    y = per_data.groupby(['age']).mean()['TenYearCHD'].values
    plt.scatter( x , y , color = "mediumblue")
    plt.xlabel("Age", alpha=0.8)
    plt.ylabel("People with CHD" , alpha=0.8)
    plt.title("Age v/s Ten Year CHD Risk  Distribution" , alpha=0.8)
    st.pyplot()
    st.text("""
    The distribution is almost linear with some outliers.
    So, Age is a good feature for prediction.""")


# Cigarettes per day v/s Ten Year CHD Risk  Distribution
# @st.cache(suppress_st_warning=True)
def cigs_chd():
    st.subheader("Cigarettes per day v/s Ten Year CHD Risk  Distribution")

    x = sorted(per_data['cigsPerDay'].unique())     # 33 unique values
    y = per_data.groupby(['cigsPerDay']).mean()['TenYearCHD'].values

    plt.scatter( x , y , color = "mediumblue")
    plt.xlabel("Cigs Per Day", alpha=0.8)
    plt.ylabel("People with CHD" , alpha=0.8)
    plt.title("Cigarettes per day v/s Ten Year CHD Risk  Distribution" , alpha=0.8)
    st.pyplot()
    st.text("Scatter is more in the middle. So, CigsPerDay is an important feature.")


# Current Smoker v/s Ten Year CHD Risk  Distribution
# @st.cache(suppress_st_warning=True)
def csmok_chd():
    st.subheader("Current Smoker v/s Ten Year CHD Risk  Distribution")

    x = per_data['currentSmoker'].unique()     # 2 unique values
    y = per_data.groupby(['currentSmoker']).mean()['TenYearCHD'].values
    bar = plt.bar( x , y , tick_label=['Non-Smoker','Smoker'])
    bar[1].set_color("royalblue")
    bar[0].set_color("lightslategrey")
    plt.xlabel("Current Smoker", alpha=0.8)
    plt.ylabel("People with CHD" , alpha=0.8)
    plt.title("Current Smoker v/s Ten Year CHD Risk  Distribution" , alpha=0.8)
    bar_plot_style()
    st.text("""
    Although, smoking is considered as a big reason for Coronary Heart Disease ,
    but data distribution shows not much difference.
    So, it is not a good feature for prediction.""")


# Gender v/s Ten Year CHD Risk  Distribution
# @st.cache(suppress_st_warning=True)
def gen_chd():
    st.subheader("Gender v/s Ten Year CHD Risk  Distribution")

    x = per_data['male'].unique()     # 2 unique values
    y = per_data.groupby(['male']).mean()['TenYearCHD'].values
    bar = plt.bar( x , y , tick_label=['female','male'] )
    bar[1].set_color("lightslategrey")
    bar[0].set_color("royalblue")
    plt.xlabel("Gender", alpha=0.8)
    plt.ylabel("People with CHD" , alpha=0.8)
    plt.title("Gender v/s Ten Year CHD Risk  Distribution" , alpha=0.8)
    bar_plot_style()
    st.text("As per data, men are slightly more prone to heart disease.")


# BPMeds v/s Ten Year CHD Risk  Distribution
# @st.cache(suppress_st_warning=True)
def bpmed_chd():
    st.subheader("BPMeds v/s Ten Year CHD Risk  Distribution")

    x = per_data['BPMeds'].unique()     # 2 unique values
    y = per_data.groupby(['BPMeds']).mean()['TenYearCHD'].values
    bars = plt.bar( x , y , tick_label=['Do not take medicine','Takes medicine'] )
    bars[0].set_color("lightslategrey")
    bars[1].set_color("royalblue")
    plt.xlabel("BPMeds", alpha=0.8)
    plt.ylabel("People with CHD" , alpha=0.8)
    plt.title("BPMeds v/s Ten Year CHD Risk  Distribution" , alpha=0.8)
    bar_plot_style()
    st.text("As per data, people who take medicines are more prone to heart disease.")


# Prevalent Stroke v/s Ten Year CHD Risk  Distribution
# @st.cache(suppress_st_warning=True)
def pstroke_chd():
    st.subheader("Prevalent Stroke v/s Ten Year CHD Risk  Distribution")

    x = per_data['prevalentStroke'].unique()     # 2 unique values
    y = per_data.groupby(['prevalentStroke']).mean()['TenYearCHD'].values
    bars = plt.bar( x , y , tick_label=['No previous stroke','Had stroke'] )
    bars[0].set_color("lightslategrey")
    bars[1].set_color("royalblue")
    plt.xlabel("Prevalent Stroke", alpha=0.8)
    plt.ylabel("People with CHD" , alpha=0.8)
    plt.title("Prevalent Stroke v/s Ten Year CHD Risk  Distribution" , alpha=0.8)
    bar_plot_style()
    st.text("As per data, people who had stroke earlier are more prone to heart disease.")


# Hypertensive v/s Ten Year CHD Risk  Distribution
# @st.cache(suppress_st_warning=True)
def hyp_chd():
    st.subheader("Hypertensive v/s Ten Year CHD Risk  Distribution")

    x = per_data['prevalentHyp'].unique()     # 2 unique values
    y = per_data.groupby(['prevalentHyp']).mean()['TenYearCHD'].values
    bars = plt.bar( x , y , tick_label=['Not Hypertensive', 'Hypertensive'] )
    bars[0].set_color("lightslategrey")
    bars[1].set_color("royalblue")
    plt.xlabel("Hypertensive", alpha=0.8)
    plt.ylabel("People with CHD", alpha=0.8)
    plt.title("Hypertensive v/s Ten Year CHD Risk  Distribution", alpha=0.8)
    bar_plot_style()
    st.text("As per data, Hypertensive patients are more prone to heart disease.")


# Diabetes v/s Ten Year CHD Risk  Distribution
# @st.cache(suppress_st_warning=True)
def diab_chd():
    st.subheader("Diabetes v/s Ten Year CHD Risk  Distribution")

    x = per_data['diabetes'].unique()     # 2 unique values
    y = per_data.groupby(['diabetes']).mean()['TenYearCHD'].values
    bars = plt.bar( x , y , tick_label=['Not Diabetic','Diabetic'] )
    bars[0].set_color("lightslategrey")
    bars[1].set_color("royalblue")
    plt.xlabel("Diabetes", alpha=0.8)
    plt.ylabel("People with CHD", alpha=0.8)
    plt.title("Diabetes v/s Ten Year CHD Risk  Distribution", alpha=0.8)
    bar_plot_style()
    st.text("As per data, Diabetic patients are more prone to heart disease.")


# Total Cholestrol v/s Ten Year CHD Risk  Distribution
# @st.cache(suppress_st_warning=True)
def totchol_chd():
    st.subheader("Total Cholestrol v/s Ten Year CHD Risk  Distribution")
    g = sns.catplot(x="TenYearCHD", y="totChol", palette="Blues", data=per_data, kind='box')
    g.set_xticklabels(["Without CHD", "With CHD"])
    g.set_axis_labels("Ten Year CHD", "Total Cholestrol")
    st.pyplot()
    st.text("As per data, people with high cholestrol are prone to heart disease.")


# Systolic Blood Pressure v/s Ten Year CHD Risk  Distribution
# @st.cache(suppress_st_warning=True)
def sysbp_chd():
    st.subheader("Systolic Blood Pressure v/s Ten Year CHD Risk  Distribution")
    g = sns.catplot(x="TenYearCHD", y="sysBP", palette="Blues", data=per_data, kind='box')
    g.set_xticklabels(["Without CHD", "With CHD"])
    g.set_axis_labels("Ten Year CHD", "Systolic Blood Pressure")
    st.pyplot()
    st.text("As per data, people with high systolic blood pressure are prone to heart disease.")


# Diastolic Blood Pressure v/s Ten Year CHD Risk  Distribution
# @st.cache(suppress_st_warning=True)
def diabp_chd():
    st.subheader("Diastolic Blood Pressure v/s Ten Year CHD Risk  Distribution")
    g = sns.catplot(x="TenYearCHD", y="diaBP", palette="Blues", data=per_data, kind='box')
    g.set_xticklabels(["Without CHD", "With CHD"])
    g.set_axis_labels("Ten Year CHD", "Diastolic Blood Pressure")
    st.pyplot()
    st.text("As per data, people with high diastolic blood pressure are prone to heart disease.")


# ---------------------------------------------------------------------------------------------------------------------
# Call to all of the functions
app_descrip()
data_understanding()
gender_dist()
age_dist()
chd_dist()
gen_cig()
heart_chol()
cigs_chol()
dist_all_data()
mean_sd()
normal_dist()
age_chd()
cigs_chd()
csmok_chd()
gen_chd()
bpmed_chd()
pstroke_chd()
hyp_chd()
diab_chd()
totchol_chd()
sysbp_chd()
diabp_chd()

# Conclusions
st.header("Conclusions")

st.text("1. Men seem to be more susceptible to heart disease than women.")
st.text("""
2. Increase in Age, systolic and diastolic Blood Pressure also 
show increasing odds of having heart disease.""")
st.text("""
3. People having medical history of Diabetes, Cholestrol, Hypertensivity,
Prevalent stroke also show increasing 
odds of having heart disease.""")
st.text("4. Glucose, BMI, Heart Rate causes a very negligible change in odds")