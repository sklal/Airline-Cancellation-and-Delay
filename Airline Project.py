# Libraries Used
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the test and train data

import os
dirpath = os.getcwd()
print("current directory is : " + dirpath)
# Train
filepath1 = os.path.join(os.getcwd(), '2018.csv')
print(filepath1)
airline = pd.read_csv(filepath1, encoding="ISO-8859-1")

# Basic Checks


def dfChkBasics(dframe):
    cnt = 1

    try:
        print(str(cnt)+': info(): ')
        cnt += 1
        print(dframe.info())
    except:
        pass

    print(str(cnt)+': describe(): ')
    cnt += 1
    print(dframe.describe())

    print(str(cnt)+': dtypes: ')
    cnt += 1
    print(dframe.dtypes)

    try:
        print(str(cnt)+': columns: ')
        cnt += 1
        print(dframe.columns)
    except:
        pass

    print(str(cnt)+': head() -- ')
    cnt += 1
    print(dframe.head())

    print(str(cnt)+': shape: ')
    cnt += 1
    print(dframe.shape)


def dfChkValueCnts(dframe):
    for i in dframe.columns:
        print(dframe[i].value_counts())


dfChkBasics(airline)
dfChkValueCnts(airline)

airline.columns

# Standardizing Column Names
airline.rename(columns={'FL_DATE': 'Date of Flight',
                        'OP_CARRIER': 'Airline Carrier',
                        'OP_CARRIER_FL_NUM': 'Flight Number',
                        'ORIGIN': 'Airport Departure Code',
                        'DEST': 'Airport Arrival Code',
                        'CRS_DEP_TIME': 'Planned Departure Time',
                        'DEP_TIME': 'Actual Departure Time',
                        'DEP_DELAY': 'Delay Time Departure',
                        'CRS_ARR_TIME': 'Planned Arrival Time',
                        'ARR_TIME': 'Actual Arrival Time',
                        'CRS_ELAPSED_TIME': 'Planned Elapsed Time',
                        'ARR_DELAY': 'Delay Time Arrival'


                        }, inplace=True)

airline.columns = airline.columns.str.replace('_', ' ')
airline.columns = map(str.title, airline.columns)
airline.columns

# Dropping the Column 'Unnamed: 27'

airline.drop(['Unnamed: 27'], axis=1, inplace=True)

# Check unique values in OP_CARRIER (airline) column
airline['Airline Carrier'].unique()

# Renaming the Carriers

airline['Airline Carrier'].replace({
    'UA': 'United Airlines',
    'AS': 'Alaska Airlines',
    '9E': 'Endeavor Air',
    'B6': 'JetBlue Airways',
    'EV': 'ExpressJet',
    'F9': 'Frontier Airlines',
    'G4': 'Allegiant Air',
    'HA': 'Hawaiian Airlines',
    'MQ': 'Envoy Air',
    'NK': 'Spirit Airlines',
    'OH': 'PSA Airlines',
    'OO': 'SkyWest Airlines',
    'VX': 'Virgin America',
    'WN': 'Southwest Airlines',
    'YV': 'Mesa Airline',
    'YX': 'Republic Airways',
    'AA': 'American Airlines',
    'DL': 'Delta Airlines'
}, inplace=True)

airline['Airline Carrier'].unique()

# Converting Delay time into hours
airline['Delay Time Arrival'] = airline['Delay Time Arrival'] / 60
airline['Delay Time Departure'] = airline['Delay Time Departure'] / 60

# Convert Date Of Flight Data Type

pd.to_datetime(airline['Date Of Flight'])

# Extracting month variable from the dataset
airline['Month'] = pd.to_datetime(airline['Date Of Flight']).dt.month
# Extracting Day variable from the dataset
airline['Day'] = pd.to_datetime(airline['Date Of Flight']).dt.weekday_name
