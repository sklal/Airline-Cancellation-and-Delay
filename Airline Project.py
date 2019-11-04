# Libraries Used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

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
    'UA': 'United airline',
    'AS': 'Alaska airline',
    '9E': 'Endeavor Air',
    'B6': 'JetBlue Airways',
    'EV': 'ExpressJet',
    'F9': 'Frontier airline',
    'G4': 'Allegiant Air',
    'HA': 'Hawaiian airline',
    'MQ': 'Envoy Air',
    'NK': 'Spirit airline',
    'OH': 'PSA airline',
    'OO': 'SkyWest airline',
    'VX': 'Virgin America',
    'WN': 'Southwest airline',
    'YV': 'Mesa Airline',
    'YX': 'Republic Airways',
    'AA': 'American airline',
    'DL': 'Delta airline'
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

# Formatting Time


def conv_time(time_val):
    if pd.isnull(time_val):
        return np.nan
    else:
            # replace 24:00 o'clock with 00:00 o'clock:
        if time_val == 2400:
            time_val = 0
            # creating a 4 digit value out of input value:
        time_val = "{0:04d}".format(int(time_val))
        # creating a time datatype out of input value:
        time_formatted = datetime.time(int(time_val[0:2]), int(time_val[2:4]))
    return time_formatted


airline['Actual Arrival Time'] = airline['Actual Arrival Time'].apply(
    conv_time)
airline['Actual Departure Time'] = airline['Actual Departure Time'].apply(
    conv_time)
airline['Planned Departure Time'] = airline['Planned Departure Time'].apply(
    conv_time)
airline['Wheels On'] = airline['Wheels On'].apply(conv_time)
airline['Wheels Off'] = airline['Wheels Off'].apply(conv_time)
airline['Planned Arrival Time'] = airline['Planned Arrival Time'].apply(
    conv_time)

# Cancellation Code Mapping
# A - Airline/Carrier - 1
# B - Weather - 2
# C - National Air System - 3
# D - Security - 4
# Not Applicable - 0

airline.loc[airline['Cancellation Code'] == 'A', 'Cancellation Code'] = 1
airline.loc[airline['Cancellation Code'] == 'B', 'Cancellation Code'] = 2
airline.loc[airline['Cancellation Code'] == 'C', 'Cancellation Code'] = 3
airline.loc[airline['Cancellation Code'] == 'D', 'Cancellation Code'] = 4
airline['Cancellation Code'] = airline['Cancellation Code'].fillna(0)
