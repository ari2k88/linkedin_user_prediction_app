#!/usr/bin/env python
# coding: utf-8



#Loading libraries

import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#Title

#st.title ("LinkedIn User Predictive App")

st.header ("Welcome! This is a Machine Learning classification app to predict whether an individual is a LinkedIn User. The model relies on six specific attributes. Make your selections in the sidebar and then click 'Predict.'")
#st.subheader ("Make your selections in the sidebar and click 'Predict'")

#setting my directory
#os.chdir("/Users/ari2k88/OneDrive/Georgetown MSBA Classes/Programming II/Final_Project_Mac")


#reading in the social media usage data and saving it into a new object named "s"
s = pd.read_csv("social_media_usage.csv")


#looking at the data dimensions
#s.shape


#looking at the first few rows in the dataframe
#s.head()


#creating a new function
def clean_sm(x):
    x = np.where(x == 1, 1,0)
    return x


#creating an empty dataframe
toy_df = pd.DataFrame({'Col1': [1,20,0],
                      'Col2': [2,1,1]})

#looking at the new dataframe
#toy_df


#testing the clean_sm function on the empty dataframe
toy_df = clean_sm(toy_df)


#looking at the updated toy_df
#toy_df


#creating a new variable "sm_li" to indicate whether a user uses LinkedIn or not
s['sm_li'] = clean_sm(s['web1h'])



#confirming the new column was appended correctly
s['sm_li'].unique()



#making a copy of the s dataframe with only the columns required for modeling
ss = s[['sm_li','income','educ2','par','marital','gender','age']].copy()


#looking at the first few rows of the new dataframe"ss"
#ss.head()


#removing observations with values considered "missing"
ss = ss[(ss["income"] <= 9)&
        (ss["educ2"] <= 8)&
        (ss["age"]<=98)]


#looking at the shape for the updated dataframe
#ss.shape


#confirming "missing" values have been removed from income variable
ss['income'].unique()


#confirming "missing" values have been removed from educ2 variable
ss['educ2'].unique()


#confirming "missing" values have been removed from age variable
ss['age'].unique()


#confirming whether there are any other null values
#ss.info()


#looking at the balance in the data
#round(ss['sm_li'].value_counts(normalize = True),2)


# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income","educ2","par","marital","gender","age"]]



# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,# hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility



# Initialize algorithm

lr = LogisticRegression()


# Fit algorithm to training data

lr.fit(X_train, y_train)


# Making predictions using the model and the testing data

y_pred = lr.predict(X_test)

#using sklearn.metrics accuracy_score

#round(accuracy_score(y_test,y_pred),2)

#confusion matrix

confusion_matrix(y_test,y_pred)


#confusion_matrix(y_test, y_pred) in pandas dataframe

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="Blues")


#calculating recall by hand, recall: TP/(TP+FN)

recall = round(36/(36+48),2)
#recall


#calculating precision by hand, precision: TP/(TP+FP)

precision = 36/(36+24)
#precision



#calculating F1 Score by hand, F1 Score = 2*(Precision * Recall)/(Precision + Recall)
f1_score = round(2*(precision*recall)/(precision+recall),2)
#f1_score



# Getting other metrics with classification_report
print(classification_report(y_test, y_pred))


# #### 10. Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?
#### Using the model to make predictions.

#- Person 1:
    #- 42 year old, female, married, non-parent, high-level of education (7), and a high income (8)
#- Person 2:
    #- 82 year old, female, married, non-parent, high-level of education (7), and a high income (8)


# #### Predicting whether Person is a LinkedIn user

#def predicted_class(x):
    #lr.predict(x)
    #return predicted_class

#def probs(x):
    #lr.predict_proba(x)
    #return probs

st.sidebar.image("LI-Logo.png", width= 200)

st.sidebar.title("Input your selections below")

with st.sidebar:
    income = st.selectbox (label = "What is your yearly income range?",
    options=["Less than $10K","$10K to under $20k",
    "$20K to under $30k","$30K to under $40k","$40K to under $50K",
    "$50K to under $75K","$75K to under $100K",
    "$100K to under $150K","Over $150K"])
    educ2 = st.selectbox(label = "What is the highest degree or level of education you have completed?",
    options =["Less than High School","High School - No Diploma",
    "High School Graduate","Some College - No Degree",
    "Two-year Associate Degree",
    "Four-year Degree",
    "Some Postgraduate - No Degree",
    "Postgraduate Degree"])
    par = st.selectbox(label= "Are you a Parent?",
    options=["Yes","No"])
    marital= st.selectbox(label = "Current marital status",
    options = ["Married","Living With a Partner","Divorced","Separated","Widowed","Never Been Married"])
    gender = st.selectbox (label = "Gender",
    options = ["Male","Female","Other"])
    age = st.slider(label = "Age",
    min_value = 1,
    max_value = 98,
    value = 35)



if income == "Less than $10K":
    income = 1
elif income == "$10K to under $20k":
    income = 2
elif income == "$20K to under $30k":
    income = 3
elif income == "$30K to under $40k":
    income = 4
elif income == "$40K to under $50K":
    income = 5
elif income == "$50K to under $75K":
    income = 6
elif income == "$75K to under $100K":
    income = 7
elif income == "$100K to under $150K":
    income = 8
elif income == "Over $150K":
    income = 9


if educ2 == "Less than High School":
        educ2 = 1
elif educ2 == "High School - No Diploma":
        educ2 = 2
elif educ2 == "High School Graduate":
        educ2 = 3
elif educ2 == "Some College - No Degree":
        educ2 = 4
elif educ2 == "Two-year Associate Degree":
        educ2 = 5
elif educ2 == "Four-year Degree":
        educ2 = 6
elif educ2 == "Some Postgraduate - No Degree":
        educ2 = 7
elif educ2 == "Postgraduate Degree":
        educ2 = 8


if par == "Yes":
       par = 1
else:
    par = 2



if marital == "Married":
        marital = 1
elif marital == "Living With a Partner":
        marital = 2
elif marital == "Divorced":
        marital = 3
elif marital == "Separated":
        marital = 4
elif marital == "Widowed":
        marital = 5
elif marital == "Never Been Married":
        marital = 6


if gender == "Male":
        gender = 1
elif gender == "Female":
        gender = 2
elif gender == "Other":
        gender = 3



    #Making predictions with model based on input vector that correspond to features used in model
# New data for features: income, education, parental status, marital status, gender, age



#st.write(income)
#st.write(educ2)
#st.write(par)
#st.write(marital)
#st.write(gender)

#st.write(type(income))
#st.write(type(educ2))
#st.write(type(par))
#st.write(type(marital))
#st.write(type(gender))
#st.write(type(age))

#person = np.array([income,educ2,par,marital,gender,age])

#person = person.reshape(1,-1)


#st.write(person)

person = [income,educ2,par,marital,gender,age]


# Predict class, given input features

if st.button ("Predict"):
   predicted_class = lr.predict([person])
   probs = lr.predict_proba([person])

   if predicted_class == 1:
    prediction = "Individual is a LinkedIn User"
   elif predicted_class == 0:
    prediction = "Individual is NOT a LinkedIn User"

    #probability = round(probs*100,2)


   st.subheader(f"Prediction: {prediction}") # 0 = not LinkedIn user; 1 = LinkedIn user
   st.subheader(f"Probability that this person is a LinkedIn User: {round(probs[0][1],2)}")




