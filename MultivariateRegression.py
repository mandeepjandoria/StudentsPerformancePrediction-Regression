import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import sklearn as sp
from math import sqrt
import seaborn as sb

# Import data from csv
MulVarData = pd.read_csv('C:/Users/Mandeep Jandoria/Desktop/StudentsPerformancePrediction/StudentsPerformance.csv', header=0)

# Data preperation
feature_cols = ['TravelTime','StudyTime', 'FreeTime', 'Health', 'MidYearGrades']
x = MulVarData[['TravelTime','StudyTime', 'FreeTime', 'Health', 'MidYearGrades']]
y = np.array(MulVarData['FinalGrades']).reshape((len(MulVarData)),1)
x = np.array(x).reshape((len(x)),5)

# Splitting data
x_train, x_test, y_train, y_test = sp.model_selection.train_test_split(x, y, random_state=1)

# Initialize and train the model
model = LinearRegression()
model.fit(x_train,y_train)

# Calculate the linear parameters
print (model.coef_)
print (model.intercept_)

# Prediction for validation set
y_pred = model.predict(x_test)
print("Root mean squared error ", sqrt(sp.metrics.mean_squared_error(y_test, y_pred)))
print ("R2 Score is", sp.metrics.r2_score(y_test, y_pred))

# Cross validation
scores = sp.model_selection.cross_val_score(model, y_test, y_pred, cv=5, scoring= 'neg_mean_squared_error')
print ("Root mean squared error for cross validation is ", sqrt(-(scores.mean())))
print ("Standard Deviation is ", scores)

sb.pairplot(MulVarData, x_vars = ['TravelTime','StudyTime', 'FreeTime', 'Health', 'MidYearGrades'], y_vars = 'FinalGrades', size=7, aspect=0.7, kind='reg')
pyplot.show()