# Importing libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Importing data files from local
h1 = pd.read_csv("C:\\Users\\DELL\\PycharmProjects\\singaporeresale\\ResaleFlatPrices19901999.csv")
h2 = pd.read_csv("C:\\Users\DELL\\PycharmProjects\\singaporeresale\\ResaleFlatPrices2000Feb2012.csv")
h3 = pd.read_csv("C:\\Users\DELL\\PycharmProjects\\singaporeresale\\ResaleFlatPricesMar2012toDec2014.csv")
h4 = pd.read_csv("C:\\Users\DELL\\PycharmProjects\\singaporeresale\\ResaleFlatPricesJan2015toDec2016.csv")
h5 = pd.read_csv("C:\\Users\DELL\\PycharmProjects\\singaporeresale\\ResaleflatpricesJan2017onwards.csv")

# concating files for processing
house = pd.concat([h1,h2,h3,h4,h5])

# processing
house.shape # check the shape and double check if all rows are uploaded or not
house.dtypes # check data types of all rows and change if required
house.isnull().sum() # check if null values are present. Impute if required
house.describe().T

# 2 different names for 1 input is corrected
house['flat_type'].value_counts()
house['flat_type'] = house['flat_type'].replace('MULTI GENERATION','MULTI-GENERATION')

house['town'].value_counts()
house['block'].value_counts()
house['street_name'].value_counts()
house['storey_range'].value_counts()
house['flat_model'].value_counts()

# Convert the 'month' column to a datetime format
house['month'] = pd.to_datetime(house['month'])

# Extract the year and month into separate columns
house['year'] = house['month'].dt.year
house['month_of_year'] = house['month'].dt.month

# Extract the year of lease commencement
house['lease_commence_year'] = pd.to_datetime(house['lease_commence_date'],format = '%Y').dt.year













