
#------------------------------------------------
#Import Libraries
#------------------------------------------------

import easygui as g
import pandas as pd

#------------------------------------------------
#Extract Data
#------------------------------------------------

regressionParams = pd.read_csv("./conf/paramsDelayed.csv")
regressionTime = pd.read_csv("./conf/paramsTime.csv")
airlines = pd.read_csv("./conf/airlines.csv")

#------------------------------------------------
# Clean Data
#------------------------------------------------

regressionParams.columns = ['Var_Name', 'Value']
regressionParams["Variables"] = 0

regressionTime.columns = ['Var_Name', 'Value']
regressionTime["Variables"] = 0

airlinesCodes = [19790,19805,19977,20304,20366,20398,20409,20416,20436,21171]
choices = ["JFK","LAX","ORD","DFW","ATL"]

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

regressionTime.loc[regressionTime['Var_Name'] == originAirport,"Variables"] = 1
regressionTime.loc[regressionTime['Var_Name'] == destinationAirport,"Variables"] = 1

#------------------------------------------------
# Regression
#------------------------------------------------

finalText = ""

for i in range(0,len(airlinesCodes)):
    regressionAirline = regressionParams.copy()
    regressionTime2 = regressionTime.copy()

    airlineTxt = "AIRLN_" + str(airlinesCodes[i])

    regressionAirline.loc[regressionAirline['Var_Name'] == airlineTxt, "Variables"] = 1
    regressionAirline["Multiplication"] = regressionAirline.apply(lambda row: row.Value * row.Variables, axis=1)

    regressionTime2.loc[regressionTime2['Var_Name'] == airlineTxt, "Variables"] = 1
    regressionTime2.loc[regressionTime2['Var_Name'] == "const", "Variables"] = 1
    regressionTime2["Multiplication"] = regressionTime2.apply(lambda row: row.Value * row.Variables, axis=1)

    finalValue = regressionAirline.Multiplication.sum() * 100
    finalTime = regressionTime2.Multiplication.sum() / 60

    airlineName = airlines.loc[airlines['Code'] == airlinesCodes[i], "Description"]
    finalText += airlineName.to_string(index=False) + " \n \t " + " Flight Time: " + str(finalTime) + " \n\n"

g.codebox("Airline Information", "Show File Contents", finalText)



