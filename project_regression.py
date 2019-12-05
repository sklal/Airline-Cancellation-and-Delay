
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500)
#%%

# Loading data and renaming columns

airline = pd.read_csv('2018.csv')
airline = pd.DataFrame(airline).loc[:50000,:]
airline.head()
airline.describe()

# Rename to make it clear
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

airline.columns = airline.columns.str.replace(' ','_')


#%%

# factorizing object type columns into numbers by method factorize

airline.Airline_Carrier,label_Carrier = pd.factorize(airline['Airline_Carrier'])
#each airline factorized according to alphabetical order

airline.Airport_Departure_Code,label_Departure = pd.factorize(airline['Airport_Departure_Code'])

airline.Airport_Arrival_Code,label_Arrival = pd.factorize(airline['Airport_Arrival_Code'])

airline.CANCELLATION_CODE,label_Cancel = pd.factorize(airline['CANCELLATION_CODE'])

airline['Month']=pd.to_datetime(airline['Date_of_Flight']).dt.month

# extract week day and factorize: 1 represents Monday - 7 represents Sunday
airline['Week'] = pd.to_datetime(airline['Date_of_Flight']).dt.weekday_name
airline['Week'],label_Week = pd.factorize(airline.Week)
airline['Week'] += 1

# Change all NaN into -100, so that we can go with regression
airline.CARRIER_DELAY=airline.CARRIER_DELAY.replace(np.nan,-100)
airline.CANCELLATION_CODE=airline.CANCELLATION_CODE.replace(np.nan,-100)
airline.NAS_DELAY=airline.NAS_DELAY.replace(np.nan,-100)
airline.WEATHER_DELAY=airline.WEATHER_DELAY.replace(np.nan,-100)
airline.SECURITY_DELAY=airline.SECURITY_DELAY.replace(np.nan,-100)
airline.LATE_AIRCRAFT_DELAY=airline.LATE_AIRCRAFT_DELAY.replace(np.nan,-100)
airline.TAXI_OUT=airline.TAXI_OUT.replace(np.nan,-100)
airline.TAXI_IN=airline.TAXI_IN.replace(np.nan,-100)


# The last column is meaningless
try:
    airline = airline.drop(['Unnamed:_27'],axis=1)
except:
    pass

# get the Y, and if look at th pattern, 2301 indicates 23:01, 
# and if you subtract two figures, what you get is hhmm, 
# so, what ever the result, hh*60+mm is the delayed time
# First turn all columns in Departure_Delay into int, so that in hundredth digit that equals to how many hours, and in tens and digits, that is how many minutes
#   There is an assumption here, we think not too many flight would delayed for a whole day, and that is extreme condition
# Second, turn the representation to array of minutes, and add that array to a new column in dataframe
airline['Departure_Delay'] = airline['Actual_Departure_Time']-airline['Planned_Departure_Time']
airline['Departure_Delay'] = airline['Departure_Delay'].replace(np.nan,99999)
airline["Departure_Delay"] = airline["Departure_Delay"].astype(int)


X=[]
for i in airline.Departure_Delay:    
    i = abs((int(i/100))*60)+((abs(i)-int(abs(i/100))*100)%100)
    X.append(i)

Xarray = np.array(X)
airline['Departure_Delay_Length'] = Xarray

#airline.Departure_Delay_Length.unique()
#%%
c = airline.Departure_Delay
countPos = 0
countNeg = 0
countZero = 0
for i in airline.Departure_Delay:
    if i>0 and i != 99999:
        countPos +=1
    elif i<0:
        countNeg +=1
    elif i==0:
        countZero +=1
print("%d flights delaied, %d flights arrived early, %d flights arrived on time" % (countPos, countNeg, countZero))

#%%
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

yDelay = airline["Departure_Delay_Length"]
xDelay = airline[["Month","Week","LATE_AIRCRAFT_DELAY","Airline_Carrier","Flight_Number","Flight_Number","TAXI_OUT","TAXI_IN","Planned_Elapsed_Time","DISTANCE","Planned_Departure_Time"]]

XtrainDelay,XtestDelay,YtrainDelay,YtestDelay = train_test_split(xDelay,yDelay)

Delay_linearmodel = linear_model.LinearRegression()
Delay_linearmodel.fit(XtrainDelay,YtrainDelay)
Delay_linearmodel.fit(XtestDelay,YtestDelay)
print('Linear model accuracy (with the test set):', Delay_linearmodel.score(XtestDelay, YtestDelay))


cv_results = cross_val_score(Delay_linearmodel, xDelay, yDelay, cv=5)
print(cv_results) 
np.mean(cv_results) 

#%%

#model building
#import statsmodels.api as sm
#from statsmodels.formula.api import glm
#model_Delay = glm(formula='Departure_Delay ~ C(Week) + C(Month) + C(Airline_Carrier) + C(Airport_Departure_Code) + Actual_Departure_Time + C(Airport_Arrival_Code) +C(CANCELLATION_CODE)', data=airline.head(1000), family = sm.families.Binomial()).fit()
#print( model_Delay.summary() )a
#
## prediction
#airline['Delay_Predict'] = model_Delay.predict(airline.head(1000))


#%%
from sklearn.model_selection import train_test_split
yDelay = airline["Departure_Delay_Length"]
xDelay = airline[["Month","Week","LATE_AIRCRAFT_DELAY","Airline_Carrier","Flight_Number","Flight_Number","TAXI_OUT","TAXI_IN","Planned_Elapsed_Time","DISTANCE","Planned_Departure_Time"]]

XtrainDelay,XtestDelay,YtrainDelay,YtestDelay = train_test_split(xDelay,yDelay)
from sklearn.linear_model import LogisticRegression
Delay_Logicmodel = LogisticRegression()
Delay_Logicmodel.fit(XtrainDelay,YtrainDelay)
Delay_Logicmodel.predict(XtestDelay)
print('Logit model accuracy (with the test set):', Delay_Logicmodel.score(XtestDelay, YtestDelay))

#%%

sns.set()
sns.pairplot(airline.iloc[:,1:])
