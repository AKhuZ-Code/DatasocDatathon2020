#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 19:51:55 2020
Title: Regression Model for Government Responses
@author: JasonKhu
"""

import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

### PREPARING THE DATASET PRIOR TO EDA AND MODELLING ###

os.getcwd()
os.chdir("/Users/JasonKhu/Desktop/Datasets")
os.getcwd()
dataset = pd.read_csv('combined_time_series.csv')

# Clean the dataset
    #remove any columns with insignificant data
dataset.isna().any() 
dataset.isna().sum()
dataset.tail()
dataset = dataset.drop(columns = ["Unnamed: 0", "retail_and_recreation_percent_change_from_baseline",
                                  "grocery_and_pharmacy_percent_change_from_baseline", 
                                  "parks_percent_change_from_baseline",
                                  "transit_stations_percent_change_from_baseline",
                                  "workplaces_percent_change_from_baseline",
                                  "residential_percent_change_from_baseline"])
dataset.columns

# Join the dataset with a populations dataset (over the date field)
    #read the population dataset
population_data = pd.read_csv('Population_data.csv')
    #merge the dataset and the population dataset
dataset = pd.merge(dataset,population_data, left_on="Country", right_on="CountryName")
dataset.tail()
dataset.columns
dataset = dataset.drop(columns=["CountryCode", "CountryName"])

#figuring out the response variable
    #let avg_to_test be the average number of days between infection and testing
time_to_policy_effect = 8

### FEATURE ENGINEERING ###

#Create percentage of population fields
    dataset["percent_pop_cases"] = 100*dataset["Cumulative Confirmed Cases"]/dataset["Population"]
    dataset["percent_pop_deaths"] = 100*dataset["Cumulative Deaths"]/dataset["Population"]
    dataset["percent_pop_recoveries"] = 100*dataset["Cumulative Recovered Cases"]/dataset["Population"]
    dataset["percent_pop_cases"]
    
# Offset the percentage fields to create response variable for our model 
response_cases = []
response_deaths = []
response_recoveries = []
rows_keep = []

dataset.shape[0]

for i in range(0,(dataset.shape[0]-1)):
    if dataset.loc[i]["Country"] == dataset.loc[i + time_to_policy_effect]["Country"]:
        response_cases.append(dataset["percent_pop_cases"].loc[i+time_to_policy_effect])
        response_deaths.append(dataset["percent_pop_deaths"].loc[i+time_to_policy_effect])
        response_recoveries.append(dataset["percent_pop_recoveries"].loc[i+time_to_policy_effect])
        rows_keep.append(i)
        
dataset = dataset.iloc[rows_keep]
dataset = dataset.reset_index()

response_cases = pd.DataFrame(response_cases)
response_deaths = pd.DataFrame(response_deaths)
response_recoveries = pd.DataFrame(response_recoveries)

dataset["response_cases"] = response_cases
dataset["response_deaths"] = response_deaths
dataset["response_recoveries"] = response_recoveries

dataset.columns
dataset.shape[0]
        
# Create variable for Days since first case of COVID-19 in each country
COVID_Days = []

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%d/%m/%Y")
    d2 = datetime.strptime(d2, "%d/%m/%Y")
    return abs((d2 - d1).days)

for i in range(0,dataset.shape[0]):
    if dataset.iloc[i]["Cumulative Confirmed Cases"] == 0:
        COVID_Days.append(0)
        day1 = dataset.iloc[i+1]["New Date"]
    else: 
        COVID_Days.append(days_between(dataset.iloc[i]["New Date"], day1))

COVID_Days = pd.DataFrame(COVID_Days)
dataset["COVID_Days"] = COVID_Days
        
# Remove rows where the number of cases is 0 to examine government responses better
dataset = dataset.drop(dataset[dataset.COVID_Days == 0].index)

# Derive unique rows from the dataset
dataset = dataset.drop_duplicates(subset = ['C1_School closing',
       'C2_Workplace closing', 'C3_Cancel public events',
       'C4_Restrictions on gatherings', 'C5_Close public transport',
       'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
       'C8_International travel controls', 'E1_Income support',
       'E2_Debt/contract relief', 'E3_Fiscal measures',
       'E4_International support', 'H1_Public information campaigns',
       'H2_Testing policy', 'H3_Contact tracing',
       'H4_Emergency investment in healthcare', 'H5_Investment in vaccines',
       'StringencyIndex', 'StringencyIndexForDisplay', 'StringencyLegacyIndex',
       'StringencyLegacyIndexForDisplay', 'GovernmentResponseIndex',
       'GovernmentResponseIndexForDisplay', 'ContainmentHealthIndex',
       'ContainmentHealthIndexForDisplay', 'EconomicSupportIndex',
       'EconomicSupportIndexForDisplay', 'COVID_Days'])
dataset = dataset.dropna()

### EDA PROCESS ###

# Extract the identifying columns of the dataset
country_identifier = dataset['Country']

# Drop all columns used for calculating columns
dataset = dataset.drop(columns = ['Country', 'Cumulative Confirmed Cases', 'New Date',
                       'Cumulative Deaths','Cumulative Recovered Cases','Population',
                       'Population'])

# Check correlations between variables

plt.figure(figsize=(20,10))
sns.heatmap(dataset.drop(columns = ['response_cases', 'response_deaths', 'response_recoveries']).corr(), annot=True) 

dataset = dataset.drop(columns = ['StringencyIndexForDisplay', 'StringencyLegacyIndexForDisplay','GovernmentResponseIndexForDisplay',
                                  'ContainmentHealthIndexForDisplay','EconomicSupportIndexForDisplay'])
dataset = dataset.drop(columns = ['StringencyIndex', 'StringencyLegacyIndex',
                                  'ContainmentHealthIndex'])

# Check correlations with the responses

dataset.drop(columns=['index','response_recoveries','response_deaths','response_cases']).corrwith(dataset.response_cases).plot.bar(figsize=(20,10),
              title = 'Correlation with Response variable',
              fontsize = 10, rot = 45,
              grid = True)

dataset.drop(columns=['index','response_recoveries','response_cases', 'response_deaths']).corrwith(dataset.response_deaths).plot.bar(figsize=(20,10),
              title = 'Correlation with Response variable',
              fontsize = 10, rot = 45,
              grid = True)

dataset.drop(columns=['index','response_cases','response_deaths', 'response_recoveries']).corrwith(dataset.response_recoveries).plot.bar(figsize=(20,10),
              title = 'Correlation with Response variable',
              fontsize = 10, rot = 45,
              grid = True)

####################################### MODELLING 1 - with cases as the response ####################################

X = dataset.drop(columns = ['response_cases', 'response_deaths', 'response_recoveries'])

# Perform training-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, dataset['response_cases'], 
    test_size = 0.2, random_state = 0)

# Training the model
from sklearn.linear_model import LinearRegression
caseRegressor = LinearRegression()
caseRegressor.fit(X_train, y_train)

from sklearn import linear_model
caseRegressor = linear_model.Lasso(alpha=0.1)
caseRegressor.fit(X_train, y_train)

from sklearn.linear_model import Ridge
caseRegressor = Ridge(alpha=0.1)
caseRegressor.fit(X_train, y_train)

# Evaluating the model
y_pred = caseRegressor.predict(X_test)
y_test

# Optimising the model
caseRegressor.score(X_test,y_test)
caseRegressor.coef_

####################################### MODELLING 2 - with deaths as the response #################################

# Perform training-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, dataset['response_deaths'], 
    test_size = 0.2, random_state = 0)

# Training the model
from sklearn.linear_model import LinearRegression
deathRegressor = LinearRegression()
deathRegressor.fit(X_train, y_train)

from sklearn import linear_model
deathRegressor = linear_model.Lasso(alpha=0.1)
deathRegressor.fit(X_train, y_train)

from sklearn.linear_model import Ridge
deathRegressor = Ridge(alpha=0.1)
deathRegressor.fit(X_train, y_train)


# Evaluating the model
y_pred = deathRegressor.predict(X_test)

# Optimising the model
deathRegressor.score(X_train,y_train)
deathRegressor.score(X_test,y_test)
deathRegressor.coef_

################################# MODELLING 3 - with recoveries as the response ####################################

# Perform training-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, dataset['response_recoveries'], 
    test_size = 0.2, random_state = 0)

# Training the model
from sklearn.linear_model import LinearRegression
recovRegressor = LinearRegression()
recovRegressor.fit(X_train, y_train)

from sklearn import linear_model
recovRegressor = linear_model.Lasso(alpha=0.1)
recovRegressor.fit(X_train, y_train)

from sklearn.linear_model import Ridge
recovRegressor = Ridge(alpha=0.1)
recovRegressor.fit(X_train, y_train)

# Evaluating the model
y_pred = recovRegressor.predict(X_test)

# Optimising the model
recovRegressor.score(X_train,y_train)
recovRegressor.score(X_test,y_test)
recovRegressor.coef_

################################# Final Remarks ####################################

#Suspected significant variables for presentation
dataset.columns[11]
dataset.columns[25]
dataset.columns[26]
