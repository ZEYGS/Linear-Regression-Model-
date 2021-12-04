#Importing the necessary packages
import pandas as pd
import numpy as np
from scipy import stats

import researchpy
import matplotlib.pyplot as plt

#Importing the SP500 stock index price and unemployment rate data 
unemployment_data = pd.read_csv("unemployment_data.csv")
print (unemployment_data)

#Correlation test to find out whether there is a correlation between X and Y
#X = unemployment rate and Y = sp500 stock index price
#Null hypothesis: Correlation between the two variables = 0
#Alternative hypothesis: Correlation between the two variables != 0 
cor_test = researchpy.ttest(unemployment_data['unemployment'], unemployment_data['sp500'])
print(cor_test)

#Calculating descriptive statistics for SP500 stock index price and unemployment rate  
#Calculating mean and median for SP500 stock index price and unemployment rate 
unemployment_mean = unemployment_data['unemployment'].mean()
unemployment_median = unemployment_data['unemployment'].median()
print ("\n\nUnemployment Rate & Sp500 Index Price Measures Of Central Tendency:")
print ('Mean Of Unemployment Rate (%): ' + str(unemployment_mean))
print ('Median Of Unemployment Rate (%): ' + str(unemployment_median))

sp500_mean = unemployment_data['sp500'].mean()
sp500_median = unemployment_data['sp500'].median()
print ('\nMean Of SP500 Index Prices ($): ' + str(sp500_mean))
print ('Median Of SP500 Index Prices ($): ' + str(sp500_median))


#Calculating min and max of range in SP500 stock index price and unemployment rate 
unemployment_min = unemployment_data['unemployment'].min()
unemployment_max = unemployment_data['unemployment'].max()
print ("\n\nUnemployment Rate & Sp500 Index Price Measures Of Spread:")
print ('Minimum Value Of Unemployment Rate (%): ' + str(unemployment_min))
print ('Maximum Value Of Unemployment Rate (%): ' + str(unemployment_max))

sp500_min = unemployment_data['sp500'].min()
sp500_max = unemployment_data['sp500'].max()
print ('\nMinimum Value Of SP500 Index Prices ($): ' + str(sp500_min))
print ('Maximum Value Of SP500 Index Prices ($): ' + str(sp500_max))


#Change of dataframe in order to graph variable rate of change over years
#Separated month and year datetimes from original Month column
#Removed original Month column
unemployment_data[["year", "month"]] = unemployment_data["Month"].str.split("-", expand = True)
print("\n\nChange Of DataFrame:")
unemployment_data = unemployment_data.drop('Month', 1)
unemployment_data = unemployment_data[["month", "year", "unemployment", "sp500"]]
print(unemployment_data)


#Computing measurement Of skewness for SP500 stock index price and unemployment rate
print("\n\nMeasurement Of Skewness For Unemployment Rate & SP500 Index Price:")
print(unemployment_data[["unemployment", "sp500"]].skew())


#Computing variance-covariance matrix of SP500 stock index price and unemployment rate
cov_matrix = unemployment_data.cov()
print("\n\nVariance-Covariance Matrix Of Unemployment Rate & SP500 Index Price:")
print(cov_matrix)


#Visualizing unemployment rate over time (years) in line graph
plt.plot(unemployment_data['year'], unemployment_data['unemployment'], color='black', marker='o', markerfacecolor='red')
plt.title('Unemployment Rate Vs Year', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Unemployment Rate (%)', fontsize=14)
plt.rcParams["figure.figsize"] = (14, 10)
plt.grid(True)
plt.show()

#Visualizing rate of SP500 Index Price over time (years) in line graph
plt.plot(unemployment_data['year'], unemployment_data['sp500'], color='black', marker='o', markerfacecolor='green')
plt.title('SP500 Index Price Vs Year', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('SP500 Index Price ($)', fontsize=14)
plt.rcParams["figure.figsize"] = (14, 10)
plt.grid(True)
plt.show()

#Computing histogram of SP500 Index Price data to further examine regression model results
unemployment_data['sp500'].plot(kind='hist',color='green', edgecolor='black')
