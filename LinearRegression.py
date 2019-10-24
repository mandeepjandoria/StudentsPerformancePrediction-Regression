import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  cross_validate
import sklearn as sp
from math import sqrt

# Import data from csv
data = pd.read_csv('C:/Users/Mandeep Jandoria/Desktop/StudentsPerformancePrediction/StudentsPerformance.csv', header=0)

# Prepare the data
x = np.array(data.MidYearGrades).reshape((len(data.MidYearGrades), 1))
y = np.array(data.MidYearGrades)

# Splitting the data
x_train, x_test, y_train, y_test = sp.model_selection.train_test_split(x,y, random_state=1)

# Initialize and train the model
model = LinearRegression()
model.fit(x_train,y_train)

# Calculate linear parameters of the model
print ("coef_ ", model.coef_)
print ("intercept_ ", model.intercept_)
y_pred = model.predict(x_test)

print("Root mean squared error ", sqrt(sp.metrics.mean_squared_error(y_test,y_pred)))
print ("R2 Score is ", sp.metrics.r2_score(y_test,y_pred))

# Cross validation with 5 folds
scores = sp.model_selection.cross_val_score(model, x_train, y_train, cv=5, scoring= 'neg_mean_squared_error')
print ("Root mean squared error for Cross Validation is ", sqrt(-(scores.mean())))

pyplot.scatter(x,y,color='black',label='Initial Data')
pyplot.plot(x,model.predict(x),color='blue',label='Linear Regression Model')
pyplot.xlabel('MidYearGrades')
pyplot.ylabel('FinalGrades')
pyplot.title('Linear Regression')
pyplot.legend()
pyplot.show()