# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 18:27:36 2022

@author: Admin
"""

import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
#warnings.filterwarnings('ignore')

df = pd.read_excel(r"C:\Users\Admin\project2\final_data.xlsx")
#df.head().style.backround_gradient('turbo')

df.drop(['Unnamed: 0'], axis = 1, inplace = True)
df.columns
df.info()
df.head()

df.rename(columns={'shortest distance Agent-Pathlab(m)' : 'Distance Agent-Pathlab', ##unit = meters
                   'shortest distance Patient-Pathlab(m)' : 'Distance Patient-Pathlab',  ##unit = meters
                   'shortest distance Patient-Agent(m)' : 'Distance Patient-Agent',  ##unit = meters
                   'Availabilty time (Patient)' : 'Patient Availabilty',  ##range format
                   'Test Booking Date' : 'Booking Date',  
                   'Test Booking Time HH:MM' : 'Booking Time',
                   'Way Of Storage Of Sample' : 'Specimen Storage',
                   ' Time For Sample Collection MM' : 'Specimen collection Time',
                   'Time Agent-Pathlab sec' : 'Agent-Pathlab sec',
                   'Agent Arrival Time (range) HH:MM' : 'Agent Arrival Time',
                   'Exact Arrival Time MM' : 'Exact Arrival Time'   ##output time
                  }, inplace=True)
df.columns
df.duplicated().any()

df.isna().any()

sns.distplot(df['Exact Arrival Time'])
ID_columns = df[['Patient ID', 'Agent ID', 'pincode']]
numerical_columns = df[['Age', 'Distance Agent-Pathlab', 'Distance Patient-Pathlab', 'Distance Patient-Agent', 
                        'Specimen collection Time' , 'Agent-Pathlab sec', 'Exact Arrival Time']]

categorical_columns = df[['patient location', 'Diagnostic Centers', 'Time slot', 'Patient Availabilty', 'Gender', 
                          'Booking Date', 'Specimen Storage', 'Sample Collection Date', 'Agent Arrival Time']]

numerical_columns.info()
categorical_columns.head()

list(categorical_columns['Diagnostic Centers'].unique())
categorical_columns['Diagnostic Centers'].value_counts().plot(kind = 'bar')
 
def name_change(text):
    if text == 'Medquest Diagnostics Center' or text == 'Medquest Diagnostics':
        return 'Medquest Diagnostics Center'
    elif text == 'Pronto Diagnostics' or text == 'Pronto Diagnostics Center':
        return 'Pronto Diagnostics Center'
    elif text == 'Vijaya Diagonstic Center' or text == 'Vijaya Diagnostic Center':
        return 'Vijaya Diagnostic Center'
    elif text == 'Viva Diagnostic' or text == 'Vivaa Diagnostic Center':
        return 'Vivaa Diagnostic Center'
    else:
        return text
    
categorical_columns['Diagnostic Centers'] = categorical_columns['Diagnostic Centers'].apply(name_change)
categorical_columns['Diagnostic Centers'].value_counts().plot(kind = 'bar')

categorical_columns['Time slot'].value_counts().plot(kind = 'bar')

categorical_columns['Specimen Storage'].value_counts().plot(kind = 'bar')

len(categorical_columns['Patient Availabilty'].unique())

categorical_columns['Patient Availabilty'].value_counts().plot(kind = 'bar')
len(categorical_columns['Agent Arrival Time'].unique())

categorical_columns['Gender'].value_counts().plot(kind = 'bar')

new_df = pd.concat([ID_columns,
                    categorical_columns[['Diagnostic Centers', 'Time slot', 'Patient Availabilty', 'Gender',
                                         'Specimen Storage', 'Agent Arrival Time']],
                    numerical_columns[['Distance Patient-Agent', 'Specimen collection Time', 'Exact Arrival Time']]
                   ], axis = 1)

new_df.info()

final = new_df[new_df['Distance Patient-Agent'] != 0]
final.info()

sns.distplot(np.log(final['Distance Patient-Agent']))

for col in final.columns[:]:
    print(col, ' : ', len(final[col].unique()), 'Unique Values')
    
    
final.describe()    
final.drop(['Patient ID', 'pincode'], axis = 1, inplace = True)
final['Distance Patient-Agent'] = np.log(final['Distance Patient-Agent'])
final = final[final['Patient Availabilty'] != '19:00 to 22:00']

#Model Buliding

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC 

#final['Patient Availabilty']
final['Patient Availabilty From'] = final['Patient Availabilty'].apply(lambda x:x.split(':')[0])
a = final['Patient Availabilty'].apply(lambda x:x.split('to')[1])
final['Patient Availabilty To'] = a.apply(lambda x:x.split(':')[0])
b = final['Agent Arrival Time'].apply(lambda x:x.split('to')[1])
final['Agent Arrive Before'] = b.apply(lambda x:x.split(':')[0])
final['Patient Availabilty From'] = final['Patient Availabilty From'].astype('int64')
final['Patient Availabilty To'] = final['Patient Availabilty To'].astype('int64')
final['Agent Arrive Before'] = final['Agent Arrive Before'].astype('int64')
final1 = final.drop(['Patient Availabilty', 'Agent Arrival Time', 'Diagnostic Centers'], axis = 1)
final1.head()
final1.columns


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
final1['Time slot'] = le.fit_transform(final1['Time slot'])
final1['Gender'] = le.fit_transform(final1['Gender'])
final1['Specimen Storage'] = le.fit_transform(final1['Specimen Storage'])
final1.head(3)

variables = final1.drop(['Exact Arrival Time'], axis = 1)
target = final1[['Exact Arrival Time']]
xtrain, xtest, ytrain, ytest = train_test_split(variables, target, test_size=0.3)
lr = LogisticRegression(multi_class='ovr')
lr.fit(xtrain, ytrain)
ypred = lr.predict(xtest)
print('Accruacy score: {:.4f}'.format(accuracy_score(ytest, ypred))) 
print('Classification Report: \n', classification_report(ytest, ypred))

#Final Model

lr1 = LogisticRegression(multi_class='ovr',
                           penalty = 'l2',
                           solver='newton-cg',
                           C = 16.0,
                           fit_intercept=True,
                           class_weight='balanced',
                           random_state=50
                          ) 
lr1.fit(xtrain, ytrain)
ypred = lr1.predict(xtest)
print('Accruacy score: {:.4f}'.format(accuracy_score(ytest, ypred))) 
print('Classification Report: \n', classification_report(ytest, ypred))

#pickle.dump(lr1, open('logistic_reg.pkl', 'wb'))
#pickle.dump(final1, open('dataset.pkl', 'wb'))


import pickle
pickle.dump(lr1, open('model_final.pkl', 'wb'))

# load the model from disk
lr1 = pickle.load(open('model_final.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(final1.iloc[0:1,:9])
list_value

print(lr1.predict(list_value))

"""from haversine import haversine
lat1 = float(input('latitude 1 :'))
lon1 = float(input('longitude 1 :'))
lat2 = float(input('latitude 2 :'))
lon2 = float(input('longitude 2 :'))

loc1 = (lat1, lon1)
loc2 = (lat2, lon2)
distance = haversine(loc1, loc2, unit='m')
print(int(distance))"""
    


























