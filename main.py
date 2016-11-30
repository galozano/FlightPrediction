#------------------------------------------------------------------------------------------------
# Flight time and delay predictor
# Description: Simple python script that runs a regression to predic actual flight time
#              and probability of delay of airlines by route 
# Creation Date: Nov 23, 2016 
#------------------------------------------------------------------------------------------------

#------------------------------------------------
# Import Libraries
# Here we import all of the necessary libraries 
#------------------------------------------------

import pandas as pd
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
import math
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------
# Extract Data from the CSV downloaded from the BOT Website
#------------------------------------------------

gdf = pd.read_csv("./CSV/merged.csv")
#list(gdf.columns.values) #Lists all of the columns names

#------------------------------------------------
# Select data to use -- We select the columns we are going to use and filter by the major airports
#------------------------------------------------

#df1 = gdf[['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'AIRLINE_ID', 'FL_NUM', 'ORIGIN', 'DEST', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP','ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP','CANCELLED', 'CANCELLATION_CODE', 'DIVERTED','ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP']]

#Select the columns we need for the analysis 
# We use these columns because in prevoius anylysis we found that are the more 
df1 = gdf[['AIRLINE_ID','ORIGIN', 'DEST', 'DEP_TIME','ARR_TIME','DEP_DELAY','ARR_DELAY','CANCELLED','DIVERTED','ACTUAL_ELAPSED_TIME']]

#Filter by the airports we want to concentrate the analysis on (ATL, DFW, JFK, LAX and ORD)
#We only used the most important airports
df2 = df1.query('(ORIGIN == "ATL" or ORIGIN == "DFW" or ORIGIN == "JFK" or ORIGIN == "LAX" or ORIGIN == "ORD") and (DEST == "ATL" or DEST == "DFW" or DEST == "JFK" or DEST == "LAX" or DEST == "ORD")')

#------------------------------------------------
#Get Random Sample Data from the data
#------------------------------------------------

#Get a 10,000 sample data 
sampledf = df2.sample(n=10000)

#------------------------------------------------
# Clean Data
#------------------------------------------------

#Trim the string columns to avoid any unexpected error
sampledf["AIRLINE_ID"] = sampledf.apply(lambda row: str.strip(str(row.AIRLINE_ID)), axis=1)
sampledf["ORIGIN"] = sampledf.apply(lambda row: str.strip(str(row.ORIGIN)), axis=1)
sampledf["DEST"] = sampledf.apply(lambda row: str.strip(str(row.DEST)), axis=1)

#Delete any rows with null values
sampledf = sampledf.dropna()

#Change Actual Elapse Time to an integer
sampledf["ACTUAL_ELAPSED_TIME"] = sampledf.apply(lambda row: int(float(row.ACTUAL_ELAPSED_TIME)), axis=1)

#Clean invalid Data - any flight that has negative time
sampledf = sampledf[sampledf.ACTUAL_ELAPSED_TIME >= 0]

#------------------------------------------------
# Add new Columns
#------------------------------------------------

#Calculate flight periords Columns - Morning is from 6 to 12 , Afternoon is from 12 to 19, Night is from 19 to 24, and Dawn is from 24 to 6
sampledf["Morning"]   = sampledf.apply(lambda row: 1 if(not row.CANCELLED and int(row.DEP_TIME) >= 600 and int(row.DEP_TIME) < 1200) else 0, axis=1)
sampledf["Afternoon"] = sampledf.apply(lambda row: 1 if(not row.CANCELLED and int(row.DEP_TIME) >= 1200 and int(row.DEP_TIME) < 1900) else 0, axis=1)
sampledf["Night"]     = sampledf.apply(lambda row: 1 if(not row.CANCELLED and int(row.DEP_TIME) >= 1900 and int(row.DEP_TIME) < 2400) else 0, axis=1)
sampledf["Dawn"]      = sampledf.apply(lambda row: 1 if(not row.CANCELLED and int(row.DEP_TIME) >= 2400 and int(row.DEP_TIME) < 600) else 0, axis=1)

#Calculate Delayed Column - Calculates if a flight was delayed or not, consideres cancelled, diverted, or delay time over 10 min a delay
sampledf["Delayed"] = sampledf.apply(lambda row: 1 if(row.CANCELLED or row.DIVERTED or row.ARR_DELAY > 30) else 0 , axis=1)

#------------------------------------------------
# Dummy Variables
#------------------------------------------------

#Create dummy variables for each relevant column
originDummy = pd.get_dummies(sampledf["ORIGIN"], prefix="ORG", drop_first=True)
destDummy = pd.get_dummies(sampledf["DEST"], prefix="DST", drop_first=True)
airlineDummy = pd.get_dummies(sampledf["AIRLINE_ID"], prefix="AIRLN", drop_first=True)

#Create a table for the regression by concatenating all of the dummy columns and the dependant variable 
dummyDf = pd.DataFrame()
dummyDf = pd.concat([originDummy,destDummy,airlineDummy,sampledf['Morning'], sampledf['Afternoon'], sampledf['Night'],sampledf['Delayed'],sampledf['ACTUAL_ELAPSED_TIME']], axis=1)

#------------------------------------------------
# Split Test & Learn Datasets
#------------------------------------------------

#Split the sample data in training and test data set -- Test size is 20% of the hole data set
trainingDF, testDF = train_test_split(dummyDf, test_size = 0.2)
#len(testDF.axes[0])

#Make sure all variables are an integer for the regression
trainingDF = trainingDF.applymap(np.int)
testDF = testDF.applymap(np.int)

#------------------------------------------------
# 50-50 Data - Divide data to have 50% delayed rows and 50% non delayed rows
#------------------------------------------------

#Get 500 rows delayed and non-delayed for the training set
trainingDFDelayed =  trainingDF[trainingDF.Delayed == 1].head(500)
trainingDFNotDelayed = trainingDF[trainingDF.Delayed == 0].head(500)

#Merge the two data sets
allTraining = [trainingDFNotDelayed,trainingDFDelayed]
trainingDF = pd.concat(allTraining)

#Get 100 rows delayed and non-delayed for the testing set
testDFDelayed = testDF[testDF.Delayed == 1].head(100)
testDFNotDelayed = testDF[testDF.Delayed == 0].head(100)

#Merge the two data sets
allTest = [testDFDelayed,testDFNotDelayed]
testDF = pd.concat(allTest)

#------------------------------------------------
# Regression - Delayed
#------------------------------------------------

#Run the regression to predict the delayed flights
XValues = sm.add_constant(trainingDF[trainingDF.columns.difference(['Delayed','ACTUAL_ELAPSED_TIME'])], prepend=False)
resultDelayed = sm.OLS(trainingDF['Delayed'], XValues).fit()
resultDelayed.summary()

#------------------------------------------------
# Regression - Predicted Total Time (Flight time + taxi)
#------------------------------------------------

#Run the regression to predict total time of flights
XValues = sm.add_constant(trainingDF[trainingDF.columns.difference(['Delayed','ACTUAL_ELAPSED_TIME'])])
resultTime = sm.OLS(trainingDF['ACTUAL_ELAPSED_TIME'], XValues ).fit()
resultTime.summary()

#------------------------------------------------
# Output Prediction Data to CSV
#------------------------------------------------

#Output of sample data
resultTime.params.to_frame().to_csv(path_or_buf="./conf/paramsTime.csv", sep=',')
resultDelayed.params.to_frame().to_csv(path_or_buf="./conf/paramsDelayed.csv", sep=',')

#---------------------------------------------------------------------------------------------------------------
#-----------VALIDATION AND TESTING -----------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

#------------------------------------------------
# Validate with Test Data -- Delayed Prediction
#------------------------------------------------

#Copy of the testing data set
validateDataDelay = testDF.copy()

#Get a subset of the data without the validation data 
subsetPredictDelay = validateDataDelay[validateDataDelay.columns.difference(['Delayed','ACTUAL_ELAPSED_TIME'])]

#Predict the outcome with the regression and put the result in a new column
subsetPredictDelay['Calculated_Delay'] = subsetPredictDelay.apply(lambda row: (row * resultDelayed.params).sum(),axis=1)

#Add the real outcome in a new column
subsetPredictDelay["Real_Delayed"] = testDF["Delayed"]

#------------------------------------------------
# Validate with Test Data -- Predicted Total Time (Flight time + taxi)
#------------------------------------------------

#Copy of the testing data set
validateDataTime = testDF.copy()

subsetPredictTime = validateDataTime[validateDataTime.columns.difference(['Delayed','ACTUAL_ELAPSED_TIME'])]
#subsetPredictTime["Calculated"] = resultTime.predict(sm.add_constant(subsetPredict))
subsetPredictTime["const"] = 1
subsetPredictTime['Calculated'] = subsetPredictTime.apply(lambda row: (row * resultTime.params).sum(),axis=1)


subsetPredictTime["ACTUAL_ELAPSED_TIME"] = validateDataTime["ACTUAL_ELAPSED_TIME"]
subsetPredictTime["Difference"] = subsetPredictTime.apply(lambda row: abs(row.ACTUAL_ELAPSED_TIME - row.Calculated), axis=1)

resultDelayed.params
subsetPredictTime.columns
subsetPredictTime.head()
#------------------------------------------------
# Calculate ROC -- Predicted Total Time (Testing set is used here)
#------------------------------------------------

#Create dataframe with the difference ranges
roicTime = pd.DataFrame({"Values":range(int(subsetPredictTime["Difference"].min()),int(subsetPredictTime["Difference"].max()),10)})
roicTime["Percentage"] = roicTime.apply(lambda row: len(subsetPredictTime[subsetPredictTime.Difference < row.Values]["Difference"]) / len(subsetPredictTime["Difference"]) * 100, axis=1 )

roicTime.to_csv(path_or_buf="./CSV/test3.csv")
roicTime
plt.plot(roicTime.Values,roicTime.Percentage)
plt.show()

#------------------------------------------------
# Calculate ROC -- Predicted Delay (Testing set is used here)
#------------------------------------------------

roicDelay = pd.DataFrame({"Values": np.arange(subsetPredictDelay["Calculated_Delay"].min(),subsetPredictDelay["Calculated_Delay"].max(),0.1)})

#True Positive
roicDelay["T_P"] = roicDelay.apply(lambda row:len(subsetPredictDelay[(subsetPredictDelay.Calculated_Delay > row.Values) & (subsetPredictDelay.Real_Delayed == 1)]),axis=1)
#False Positive
roicDelay["F_P"] = roicDelay.apply(lambda row:len(subsetPredictDelay[(subsetPredictDelay.Calculated_Delay > row.Values) & (subsetPredictDelay.Real_Delayed == 0)]),axis=1)
#True Negative
roicDelay["T_N"] = roicDelay.apply(lambda row:len(subsetPredictDelay[(subsetPredictDelay.Calculated_Delay < row.Values) & (subsetPredictDelay.Real_Delayed == 0)]),axis=1)
#False Negative
roicDelay["F_N"] = roicDelay.apply(lambda row:len(subsetPredictDelay[(subsetPredictDelay.Calculated_Delay < row.Values) & (subsetPredictDelay.Real_Delayed == 1)]),axis=1)

#False Posive Ration 
roicDelay["F_P_R"] = roicDelay.apply(lambda row: row["F_P"]/(row["F_P"] + row["T_N"]),axis=1)
#Recall Ration
roicDelay["Recall"] = roicDelay.apply(lambda row: row["T_P"]/(row["T_P"] + row["F_N"]),axis=1)

roicDelay.to_csv("")
#Plot graph 

plt.plot(roicDelay["F_P_R"],roicDelay["Recall"] )
plt.xlabel("F_P_R")
plt.ylabel("Recall")
plt.title('ROC Chart')
plt.show()


