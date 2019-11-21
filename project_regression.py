
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
airline = pd.DataFrame(airline)
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


# The last column is meaningless
airline = airline.drop(['Unnamed:_27'],axis=1)

# get the Y, and if look at th pattern, 2301 indicates 23:01, 
# and if you subtract two figures, what you get is hhmm, 
# so, what ever the result, hh*60+mm is the delayed time
airline['Departure_Delay'] = airline['Actual_Departure_Time']-airline['Planned_Departure_Time']
for i in range(0,3):
    airline.Departure_Delay[i]=abs((int(airline.Departure_Delay[i]/100))*60)+((abs(airline.Departure_Delay[i])-int(abs(airline.Departure_Delay[i]/100))*100)%100)
airline.Departure_Delay.unique()

#%%

#model building
import statsmodels.api as sm
from statsmodels.formula.api import glm
model_Delay = glm(formula='Departure_Delay ~ C(Week) + C(Month) + C(Airline_Carrier) + C(Airport_Departure_Code) + Actual_Departure_Time + C(Airport_Arrival_Code) +C(CANCELLATION_CODE)', data=airline.head(1000), family = sm.families.Binomial()).fit()
print( model_Delay.summary() )

# prediction
airline['Delay_Predict'] = model_Delay.predict(airline.head(1000))


#%%
from statsmodels.formula.api import ols
model_Delay_linear = ols(formula='Departure_Delay ~ C(Week) + C(Month) + C(Airline_Carrier) + C(Airport_Departure_Code) + Actual_Departure_Time + C(Airport_Arrival_Code) +C(CANCELLATION_CODE)', data=airline.head(1000)).fit()
print( model_Delay_linear.summary() )
airline['LM_Prediction'] = model_Delay_linear.predict(airline.head(1000))


