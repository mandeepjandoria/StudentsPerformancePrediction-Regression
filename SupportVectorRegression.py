import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import datasets, linear_model
from sklearn.svm import SVR
import sklearn as sp
import seaborn as sb
from math import sqrt

# Import data from csv file
SVCData = pd.read_csv('C:/Users/Mandeep Jandoria/Desktop/StudentsPerformancePrediction/StudentsPerformance.csv', header=0)

# Initialize the model object
svr_lin = SVR(kernel='linear')

# Data preperation
x_lin = np.array(SVCData.MidYearGrades).reshape((len(SVCData.MidYearGrades), 1))
y_lin = np.array(SVCData.FinalGrades)

# Data splitting
x_train, x_test, y_train, y_test = sp.model_selection.train_test_split(x_lin, y_lin, random_state=1)

# Train the model with training data
svr_lin.fit(x_train,y_train)

# Prediction for validation set
y_pred = svr_lin.predict(x_test)
print("Root mean squared error ", sqrt(sp.metrics.mean_squared_error(y_test,y_pred)))
print ("R2 Score is", sp.metrics.r2_score(y_test,y_pred))

# Cross validation with 5 folds
scores = sp.model_selection.cross_val_score(svr_lin, x_train, y_train, cv=5, scoring= 'neg_mean_squared_error')
print ("Root mean squared error for Cross Validation is ", sqrt(-(scores.mean())))

# Poly model for experimentation
svr_poly = SVR(kernel='poly')
svr_poly.fit(x_train,y_train)

print("Root mean squared error for polynomial is ", sqrt(sp.metrics.mean_squared_error(y_test, svr_poly.predict(x_test))))
scoresPoly = sp.model_selection.cross_val_score(svr_poly, x_train, svr_poly.predict(x_train), cv=5, scoring= 'neg_mean_squared_error')
print ("Root mean squared error for Cross Validation for polynomial is ", sqrt(-(scoresPoly.mean())))

pyplot.scatter(x_lin,y_lin,color='black',label='data')
pyplot.plot(x_lin,svr_lin.predict(x_lin),color='blue',label='Linear SVR')
pyplot.xlabel('MidYearGrades')
pyplot.ylabel('FinalGrades')
pyplot.title('Support Vector Regression')
pyplot.legend()
pyplot.show()

pyplot.scatter(x_lin,y_lin,color='black',label='data')
pyplot.plot(x_lin,svr_lin.predict(x_lin),color='blue',label='Linear SVR')
pyplot.plot(x_lin,svr_poly.predict(x_lin),color='red',label='Polynomial SVR')
pyplot.xlabel('MidYearGrades')
pyplot.ylabel('FinalGrades')
pyplot.title('Support Vector Regression')
pyplot.legend()
pyplot.show()