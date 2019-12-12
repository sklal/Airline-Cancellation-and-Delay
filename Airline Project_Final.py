#%%
#  Libraries Used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# Loading the test and train data
#%%
import os
dirpath = os.getcwd()
print("current directory is : " + dirpath)
#Train
filepath1 = os.path.join(os.getcwd(), '2018.csv')
print(filepath1)
airline = pd.read_csv(filepath1, encoding="ISO-8859-1")
#%%
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
#%%
# Standardizing Column Names
airline.rename(columns={'FL_DATE': 'Flight_Date',
                        'OP_CARRIER': 'Airline_Carrier',
                        'OP_CARRIER_FL_NUM': 'Flight_Number',
                        'ORIGIN': 'Airport_Departure_Code',
                        'DEST': 'Airport_Arrival_Code',
                        'CRS_DEP_TIME': 'Planned_Departure_Time',
                        'DEP_TIME': 'Actual_Departure_Time',
                        'DEP_DELAY': 'Delay_Time_Departure',
                        'CRS_ARR_TIME': 'Planned_Arrival_Time',
                        'ARR_TIME': 'Actual_Arrival_Time',
                        'CRS_ELAPSED_TIME': 'Planned_Elapsed_Time',
                        'ARR_DELAY': 'Delay_Time_Arrival',
                        'NAS_DELAY': 'Air_System_Delay'
                        }, inplace=True)

airline.columns = airline.columns.str.replace('_', ' ')
airline.columns = map(str.title, airline.columns)
airline.columns

# Dropping the Column 'Unnamed: 27'

airline.drop(['Unnamed: 27'], axis=1, inplace=True)

# Check unique values in OP_CARRIER (airline) column
airline['Airline Carrier'].unique()

# Renaming the Carriers

#airline['Airline Carrier'].replace({
#    'UA': 'United airline',
#    'AS': 'Alaska airline',
#    '9E': 'Endeavor Air',
#    'B6': 'JetBlue Airways',
#    'EV': 'ExpressJet',
#    'F9': 'Frontier airline',
#    'G4': 'Allegiant Air',
#    'HA': 'Hawaiian airline',
#    'MQ': 'Envoy Air',
#    'NK': 'Spirit airline',
#    'OH': 'PSA airline',
#    'OO': 'SkyWest airline',
#    'VX': 'Virgin America',
#    'WN': 'Southwest airline',
#    'YV': 'Mesa Airline',
#    'YX': 'Republic Airways',
#    'AA': 'American airline',
#    'DL': 'Delta airline'
#}, inplace=True)

#%%
#airline['FL_DATE_weekday'] = pd.to_datetime(airline['FL_DATE']).dt.weekday_name
airline.head(6)
#%%
airline['Airline Carrier'].unique()



#%%
# Convert Date Of Flight Data Type

#pd.to_datetime(airline['Date Of Flight'])
#%%



# Extracting month variable from the dataset
#airline['Month'] = pd.to_datetime(airline['Date Of Flight']).dt.month
# Extracting Day variable from the dataset
#airline['Day'] = pd.to_datetime(airline['Date Of Flight']).dt.weekday_name

#%%
# Formatting Time





#%%



# Cancellation Code Mapping
# A - Airline/Carrier - 1
# B - Weather - 2
# C - National Air System - 3
# D - Security - 4
# Not Applicable - 0
#%%
airline.loc[airline['Cancellation_Code'] == 'A', 'Cancellation_Code'] = 1
airline.loc[airline['Cancellation_Code'] == 'B', 'Cancellation_Code'] = 2
airline.loc[airline['Cancellation_Code'] == 'C', 'Cancellation_Code'] = 3
airline.loc[airline['Cancellation_Code'] == 'D', 'Cancellation_Code'] = 4
#airline['Cancellation Code'] = airline['Cancellation Code'].fillna(0)
#airline['Carrier Delay'] = airline['Carrier Delay'].fillna(0)
#airline['Weather Delay'] = airline['Weather Delay'].fillna(0)
#airline['Nas Delay'] = airline['Nas Delay'].fillna(0)
#airline['Security Delay'] = airline['Security Delay'].fillna(0)
#airline['Late Aircraft Delay'] = airline['Late Aircraft Delay'].fillna(0)


#%%
airline.isnull().sum()
# %%
airline['Cancellation_Code'] = airline['Cancellation_Code'].fillna(0)
airline['Carrier_Delay'] = airline['Carrier_Delay'].fillna(0)
airline['Weather_Delay'] = airline['Weather_Delay'].fillna(0)
airline['Air_System_Delay'] = airline['Air_System_Delay'].fillna(0)
airline['Security_Delay'] = airline['Security_Delay'].fillna(0)
airline['Late_Aircraft_Delay'] = airline['Late_Aircraft_Delay'].fillna(0)
airline.isnull().sum()

# %%
airline.shape

# %%
# removing missing values as the counts are low
airline = airline.dropna(axis=0)
airline.isnull().sum()
airline.shape











#%%
airline.head()

# %%
airline['Total_Delay'] = airline['Delay_Time_Arrival'] + airline['Delay_Time_Departure']

# %%
airline.isnull().sum()

# %%
airline['totaldelaybin'] = np.where(airline['Total_Delay'] > 0 , 1, 0 )

# %%
airline.head()

# %%
#import pandas as pd
#import numpy as np
#rs = np.random.RandomState(0)
#corr = airline.corr()
#corr.style.background_gradient(cmap='coolwarm')
#airline['Airline Carrier'] = airline['Airline Carrier'].astype('str')
#airline['Airport Arrival Code'] = airline['Airport Arrival Code'].astype('str')
#airline['Planned Departure Time'] = airline['Planned Departure Time'].astype('str')
#airline['Planned Departure Time'] = airline['Planned Departure Time'].astype('str')

#%%
#airline['FL_DATE_weekday'] = pd.to_datetime(airline['Date of Flight'])

# %%
#dropping categorical data for corr plot
airline1 = airline.drop(['Airline_Carrier','Airport_Departure_Code', 'Airport_Arrival_Code','Cancelled','Cancellation_Code','Diverted','Total_Delay','Flight_Date',], axis = 1)
#airline1 = airline.drop(['Airline Carrier',], axis = 1)
#airline1 = 


#%%
airline1.isnull().sum()
# %%
rs = np.random.RandomState(0)
corr = airline1.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
#corr.to_csv("C:/DS/Introduction to data Mining/airline/Airline-Cancellation-and-Delay-master/Airline-Cancellation-and-Delay-master/corr.csv")

# %%
# from co relation matrix we are removing 
# Actual Elapsed Time
#Air Time
#Distance
#Delay Time Departure
#Delay Time Arrival
#'Planned Departure Time','Delay Time Departure','Planned Arrival Time


airline1 = airline1.drop(['Air_Time','Distance','Planned_Arrival_Time','Actual_Arrival_Time','Planned_Departure_Time','Actual_Departure_Time','Actual_Arrival_Time','Actual_Departure_Time','Delay_Time_Arrival','Planned_Elapsed_Time','Planned_Departure_Time','Delay_Time_Departure','Planned_Arrival_Time',], axis = 1)
#airline1 = airline.drop(['Origin',], axis = 1)
#%%

Y = airline1['totaldelaybin']
X = airline1.loc[:, airline1.columns != 'totaldelaybin']
#%%
#X = X.drop(['Airport Departure Code','Airport Arrival Code','Airline Carrier',], axis = 1 )
#X = X.drop(['Date Of Flight',], axis = 1 )
#X = X.drop(['Total_Delay',], axis = 1 )
#X= X.drop(['Planned Departure Time','Delay Time Departure','Planned Arrival Time',], axis = 1 )
#%%
#df['FL_DATE_weekday'] = pd.to_datetime(df['FL_DATE']).dt.weekday_name
#%%
#No. of Flights per Carrier
plt.figure(figsize=(25, 8))
bands_country = airline['Airline_Carrier'].value_counts()
plt.title('No. of flights per carrier')
sns.barplot(x=bands_country[:15].keys(),
            y=bands_country[:15].values, palette="GnBu_d")
plt.savefig('ac.png')
#%%
#Delays by Airport
airport_by_delay = airline.groupby('Airport_Departure_Code').Delay_Time_Arrival.sum().sort_values(ascending=False)
plt.figure(figsize=(20, 6))
airport_by_delay[:15].plot.bar()
plt.title('Delays by Airport')
plt.xlabel('Airport')
plt.ylabel('Hours')
plt.savefig('dap.png')

plt.show()
#%%
#Delayed flights per Carrier
airline['Delayed'] = airline.loc[:,'Delay_Time_Arrival'].values > 0
airline = airline.sort_values(['Delayed']).reset_index(drop=True)
figsize=plt.subplots(figsize=(10,12))
sns.countplot(x='Delayed',hue='Airline_Carrier',data=airline)
plt.title("Delayed Status")
plt.savefig("delayed.png")
plt.show()

#%%
#Taxiin-TaxiOut comparison
axis = plt.subplots(figsize=(20,14))
sns.set_color_codes("pastel")
sns.set_context("notebook", font_scale=1)
axis = sns.barplot(x="Taxi_Out", y="Airline_Carrier", data=airline, color="b")
axis = sns.barplot(x="Taxi_In", y="Airline_Carrier", data=airline, color="g")
plt.title("Taxi In-Taxi Out Comparison")
axis.set(xlabel="TAXI_TIME (TAXI_OUT: green, TAXI_IN: blue)")
plt.savefig("tx.png")

#%%
# test train

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)


# %%
#Model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred = logistic_regression.predict(X_test) 

#%%
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
#sns.heatmap(confusion_matrix, annot=True)
confusion_matrix

# %%
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
#airline.dtypes






# %%
