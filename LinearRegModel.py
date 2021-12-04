#Importing the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

#Importing the dataset and saving the variables
unemployment_data = pd.read_csv("unemployment_data.csv")
print(unemployment_data.head())
X1 = unemployment_data[['unemployment']].values #independent variable
X2 = unemployment_data[['disposableincome']].values #control variable
y = unemployment_data['sp500'].values #dependent variable


#Splitting the dataset into training set (75%) and testing set (25%)
np.random.seed(123) #Setting seed so we get same results every time
X_train, X_test, y_train, y_test = \
   model_selection.train_test_split(X1, y, train_size=0.75)
   
#Creating linear regression model
unemployment_model = linear_model.LinearRegression()
#Fitting the model
unemployment_model.fit(X_train, y_train)

#Interpreting the model
intercept = unemployment_model.intercept_
betacoef = unemployment_model.coef_
print('Intercept (beta_0) : {}.'.format(intercept))
print('Beta_1 : {}.'.format(betacoef))

#Predicting sp500 for the testing set from the model above
y_predicted = unemployment_model.predict(X_test)

#Computing R^2
rsqtraining = unemployment_model.score(X_train, y_train)
rsqtesting = unemployment_model.score(X_test, y_test)
print('R-squared of training set: {}.'.format(rsqtraining))
print('R-squared for the testing set: {}.'.format(rsqtesting))

#Computing RMSE (root mean squared error) for the testing set
print('RMSE for testing set: {}.'.format(np.sqrt(metrics.mean_squared_error(y_test, y_predicted))))

#Plotting regression line on the training set
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, unemployment_model.predict(X_train), color='pink')
plt.title("S&P 500 Index Price Vs. Unemployment Rate: Training set")
plt.xlabel("U.S. Unemployment Rate (%)") 
plt.ylabel("S&P 500 Index Price($)") 
plt.show()

#Plotting regression line on the testing set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_predicted, color='blue')
plt.title("S&P 500 Index Price Vs. Unemployment Rate: Testing set")
plt.xlabel("U.S. Unemployment Rate(%)") 
plt.ylabel("S&P 500 Index Price($)") 
plt.show()



#The plots show a nonlinear relationship so we need to modify our model

#Trying to improve the model with a log transformation on x and y
log_x = np.log(unemployment_data[['unemployment']].values)
log_y = np.log(unemployment_data[['sp500']].values)

#Splitting transformed data into training and testing set
np.random.seed(123)
log_x_train, log_x_test, y_train_log, y_test_log = \
    model_selection.train_test_split(log_x, log_y, train_size = 0.75)

#Creating and fitting new regression model
log_model = linear_model.LinearRegression()
log_model.fit(log_x_train, y_train_log)

#Interpreting the new model
intercept_log = log_model.intercept_
betacoef_log = log_model.coef_
print('Intercept (beta_0) after log transformation : {}.'.format(intercept_log))
print('Beta_1 after log transformation: {}.'.format(betacoef_log))

#Predicting sp500 for the testing set from the model above
y_pred_log = log_model.predict(log_x_test)

#Computing R^2  for the new model
rsqtraining = log_model.score(log_x_train, y_train_log)
rsqtesting = log_model.score(log_x_test, y_test_log)
print('R-squared of training set after log transformation: {}.'.format(rsqtraining))
print('R-squared for the testing set after log transformation: {}.'.format(rsqtesting))

#Computing RMSE (root mean squared error) for the new model
print('RMSE for testing set after log transformation: {}.'.format(np.sqrt(metrics.mean_squared_error(y_test_log, y_pred_log))))

#Plotting regression line on the training set
plt.scatter(log_x_train, y_train_log, color='blue')
plt.plot(log_x_train, log_model.predict(log_x_train), color='pink')
plt.title("Log Transformation Model: Training set")
plt.xlabel("Log of U.S. Unemployment Rate (%)") 
plt.ylabel("Log of S&P 500 Index Price($)") 
plt.show()

#Plotting regression line on the testing set
plt.scatter(log_x_test, y_test_log, color='red')
plt.plot(log_x_test, y_pred_log, color='blue')
plt.title("Log Transformation Model: Testing set")
plt.xlabel("Log of U.S. Unemployment Rate(%)") 
plt.ylabel("Log of S&P 500 Index Price($)") 
plt.show()


#Multiple regression model with control variable to account for other
#factors that may affect stock market, to try to isolate the effect of
#unemployment on the stock market

#Creating dataframe for independent variables
X = unemployment_data.drop(['sp500','Month'],axis=1)

#Creating series for response variable
Y = unemployment_data.sp500

#Scaling the independent variables since they have differing scales
scale_X = preprocessing.scale(X)
scale_X = pd.DataFrame(scale_X,columns=X.columns)

#Splitting into 75% training and 25% testing set
X_train,X_test,Y_train,Y_test = \
    model_selection.train_test_split(scale_X,Y,train_size=0.75)


#Creating linear regression model
multiple_model = linear_model.LinearRegression()
multiple_model.fit(X_train,Y_train)

#Making predictions on the testing data
multiple_pred = multiple_model.predict(X_test)

#Interpreting the model
multiple_intercept = multiple_model.intercept_
multiple_coef = multiple_model.coef_
print('Intercept (Beta0): {}.'.format(multiple_intercept))
print('Regression coefficients: {}.'.format(multiple_coef))


#Computing R^2 for multiple regression model
print('R squared for the training set: {}.'.format(multiple_model.score(X_train,Y_train)))
print('R squared for the testing set: {}.'.format(multiple_model.score(X_test,Y_test)))

#Computing RMSE for the multiple regression model
print('RMSE for testing set after log transformation: {}.'.format(np.sqrt(metrics.mean_squared_error(Y_test, multiple_pred))))