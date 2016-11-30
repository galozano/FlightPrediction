# FlightPrediction
Simple script that predicts flight time and probability of delay for local (USA) flights.

#Description of Files and Folders

main.py: Script that creates a regression model to predict flight time and possibility of delay based on the data from the bureau of transportation statistics.

app.py: Simple application to use the regression results of main.py in a simple GUI that helps the user select the origin and destination airport and obtain the flight time and probability of delay for each of the airlines.

CSV Folder: Folder with the initial dataset. The data set from the bureau of transportation statistics was used and not uploaded to git due to its size. To use the main.py script you can go to the below webpage, download the data, and uploaded it to the CSV folder.

http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236

Conf Folder: Folder where the results of the regressions are saved.
