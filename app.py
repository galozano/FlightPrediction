
#------------------------------------------------
#Import Libraries
#------------------------------------------------

import easygui as g
import pandas as pd

#------------------------------------------------
#Extract Data
#------------------------------------------------

regressionParams = pd.read_csv("./conf/paramsDelayed.csv")
airlines = pd.read_csv("./conf/airlines.csv")

#------------------------------------------------
# Clean Data
#------------------------------------------------

regressionParams.columns = ['Var_Name', 'Value']
regressionParams["Variables"] = 0
airlinesCodes = [19790,19805,19977,20304,20366,20398,20409,20416,20436,21171]
choices = ["JFK","MIA","LAX","ORD","DFW"]

#------------------------------------------------
# Ask for Airports
#------------------------------------------------

msg ="Please select origin airport"
title = "Origin airport"
originAirport = g.choicebox(msg, title, choices)
originAirport = 'ORG_' + originAirport

msg ="Please select destination airport"
title = "Destination airport"
destinationAirport = g.choicebox(msg, title, choices)
destinationAirport = 'DST_' + destinationAirport

#------------------------------------------------
# Regression Variables
#------------------------------------------------

regressionParams.loc[regressionParams['Var_Name'] == originAirport,"Variables"] = 1
regressionParams.loc[regressionParams['Var_Name'] == destinationAirport,"Variables"] = 1

#------------------------------------------------
# Regression
#------------------------------------------------

finalText = ""

for i in range(0,len(airlinesCodes)):
    regressionAirline = regressionParams.copy()
    airlineTxt = "AIRLN_" + str(airlinesCodes[i])
    regressionAirline.loc[regressionAirline['Var_Name'] == airlineTxt, "Variables"] = 1
    regressionAirline["Multiplication"] = regressionAirline.apply(lambda row: row.Value * row.Variables, axis=1)
    finalValue = regressionAirline.Multiplication.sum() * 100
    airlineName = airlines.loc[airlines['Code'] == airlinesCodes[i], "Description"]
    finalText += airlineName.to_string(index=False) + " \n \t Delay Prob:" + "%.2f" % finalValue + "% \n\n"

g.codebox("Airline Information", "Show File Contents", finalText)



