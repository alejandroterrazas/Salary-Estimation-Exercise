#!/usr/bin/env python
# coding: utf-8

##progam to run  forward pass on test data.
##uses keras neural net model to make predictions and output
##to pandas.

#output is a csv file with jobId and salary estimate.


# to run this module type:  python RunTest.py
# # Required imports
import keras
from timeit import default_timer as timer

from keras.models import model_from_json

import pandas as pd
import numpy as np
import sys

from keras.models import model_from_json
import pickle

start = timer()
json_file = open('./model.json', 'r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                                                                               

##read data TEST datasets (previously split used SplitTrainCSV.ipynb)

#df  = pd.read_csv('./TEST.csv')
fname = sys.argv[1]
print("running script for filename: ", fname)
df = pd.read_csv(fname)
ordinal_values = {"degree": {"NONE": 1, "HIGH_SCHOOL": 2,
                              "BACHELORS": 3, "MASTERS": 4,
                              "DOCTORAL": 4},                     
                  "jobType": {"JANITOR": 1, "JUNIOR": 2,
                              "SENIOR": 3, "MANAGER": 4,
                              "VICE_PRESIDENT": 5, 
                              "CTO": 6, "CFO": 7, 
                              "CEO": 8},                 
                  "industry": {"EDUCATION": 1, "AUTO": 1,
                               "SERVICE": 2, "HEALTH": 2, "WEB": 2,
                               "OIL": 3, "FINANCE": 3},
                  "major":  {"NONE": 1, "LITERATURE": 2, 
                             "BIOLOGY": 3, "CHEMISTRY": 3,
                             "PHYSICS": 4, "MATH": 4,
                             "COMPSCI": 5, "ENGINEERING": 5, "BUSINESS": 5}}
   
df.replace(ordinal_values, inplace=True)

df['distance'] = pd.cut(df.milesFromMetropolis, 
                              10, labels=list(range(10,0,-1)))

df['years'] = pd.cut(df.yearsExperience, 10, labels=list(range(1,11)))
df = df.astype({'years': 'int32', 'distance': 'int32'})


# split into input (X) and output (Y) variables

X = np.hstack([df.jobType.values.reshape(-1,1), 
               df.years.values.reshape(-1,1),
               df.degree.values.reshape(-1,1),
               df.distance.values.reshape(-1,1),
               df.industry.values.reshape(-1,1),
               df.major.values.reshape(-1,1)])


##load scalarX and scalarY from pickle file
scalarX = pickle.load(open('./xtransform.pkl', 'rb'))
scalarY = pickle.load(open('./ytransform.pkl', 'rb'))

tX = scalarX.transform(X)
y_hat_nn = model.predict(tX)

y_hat_nn = scalarY.inverse_transform(y_hat_nn)[:,0]

df_results = pd.DataFrame()
df_results['salary'] = np.round(y_hat_nn).astype(int)

df_results['jobId'] = df.jobId

print("Number of records", len(df_results))

print("Time taken: ", timer()-start)
df_results.to_csv('./test_salaries.csv', columns=['jobId', 'salary'], index=False)
