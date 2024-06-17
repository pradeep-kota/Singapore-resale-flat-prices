#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[53]:


from sklearn.preprocessing import LabelEncoder


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[8]:


h1 = pd.read_csv("C:\\Users\\DELL\\PycharmProjects\\singaporeresale\\ResaleFlatPrices19901999.csv")
h2 = pd.read_csv("C:\\Users\DELL\\PycharmProjects\\singaporeresale\\ResaleFlatPrices2000Feb2012.csv")
h3 = pd.read_csv("C:\\Users\DELL\\PycharmProjects\\singaporeresale\\ResaleFlatPricesMar2012toDec2014.csv")
h4 = pd.read_csv("C:\\Users\DELL\\PycharmProjects\\singaporeresale\\ResaleFlatPricesJan2015toDec2016.csv")
h5 = pd.read_csv("C:\\Users\DELL\\PycharmProjects\\singaporeresale\\ResaleflatpricesJan2017onwards.csv")


# In[10]:


house = pd.concat([h1,h2,h3,h4,h5])


# In[12]:


house.shape


# In[36]:


house.dtypes


# In[18]:


house.isnull().sum()


# In[20]:


house.info()


# In[21]:


house.describe().T


# In[29]:


house['flat_type'].value_counts()


# In[28]:


house['flat_type'] = house['flat_type'].replace('MULTI GENERATION','MULTI-GENERATION')


# In[34]:


house['town'].value_counts()


# In[30]:


house['block'].value_counts()


# In[31]:


house['street_name'].value_counts()


# In[32]:


house['storey_range'].value_counts()


# In[33]:


house['flat_model'].value_counts()


# In[35]:


# Convert the 'month' column to a datetime format
house['month'] = pd.to_datetime(house['month'])

# Extract the year and month into separate columns
house['year'] = house['month'].dt.year
house['month_of_year'] = house['month'].dt.month
house


# In[37]:


# Extract the year of lease commencement
house['lease_commence_year'] = pd.to_datetime(house['lease_commence_date'],format = '%Y').dt.year


# In[48]:


house.dtypes


# In[38]:


# Extract the remaining_lease feature
data = house['remaining_lease']
house_new = pd.DataFrame(data)

# Extract years and months using regular expressions
lease_info = house['remaining_lease'].str.extract(r'(\d+) years (\d+) months')
lease_info.columns = ['years', 'months']

# Convert to numeric values
house['remaining_lease_years'] = pd.to_numeric(lease_info['years'])
house['remaining_lease_months'] = pd.to_numeric(lease_info['months'])


# In[184]:


house.columns


# In[43]:


house_new_data = house.copy()


# In[50]:


house_new_data.isnull().sum()


# In[49]:


house_new_data.drop(columns=['month','block','lease_commence_date','remaining_lease'],inplace = True)


# In[51]:


# handling the null values using mean method
house_new_data['remaining_lease_years'].fillna(house['remaining_lease_years'].mean(),inplace = True)
house_new_data['remaining_lease_months'].fillna(house['remaining_lease_months'].mean(),inplace = True)
house_new_data.isnull().sum()


# In[54]:


encoder = LabelEncoder()
house_new_data['town'] = encoder.fit_transform(house_new_data['town'])
house_new_data['flat_type'] = encoder.fit_transform(house_new_data['flat_type'])
house_new_data['storey_range'] = encoder.fit_transform(house_new_data['storey_range'])
house_new_data['flat_model'] = encoder.fit_transform(house_new_data['flat_model'])


# In[55]:


house_new_data


# In[56]:


# checking the outliars using boxplot distribution plot
def plot(house_new_data,column):
    plt.figure(figsize =(15,6))
    plt.subplot(1,3,1)
    sns.boxplot(data = house_new_data ,x = column)
    plt.title(f'box plot for {column}')
    

    plt.subplot(1,3,2)
    sns.histplot(data = house_new_data ,x = column,kde = True ,bins = 40)
    plt.title(f'distribution  plot for {column}')
    plt.show()


# In[57]:


for i in ['floor_area_sqm','resale_price','lease_commence_year','remaining_lease_years']:
    plot(house_new_data ,i)


# In[58]:


# 'floor_area_sqm','resale_price' this two feature are skewd in data handling log method 
house_new_data['floor_area_sqm'] = np.log(house_new_data['floor_area_sqm'])
house_new_data['resale_price'] = np.log(house_new_data['resale_price'])
house_new_data


# In[59]:


for i in ['floor_area_sqm','resale_price']:
    plot(house_new_data,i)


# In[60]:


#using the IQR & Clips  method removing the outliars :
# formula IQR = Q3_Q1
def outlier(house_new_data ,column):
    IQR = house_new_data[column].quantile(0.75)-house_new_data[column].quantile(0.25)
    upper_value = house_new_data[column].quantile(0.75)+1.5*IQR
    lower_value = house_new_data[column].quantile(0.25)-1.5*IQR
    
    house_new_data[column] =     house_new_data[column].clip(upper_value,lower_value)

outlier(house_new_data, 'floor_area_sqm')
outlier(house_new_data, 'resale_price')
house_new_data1 = house_new_data.copy()


# In[61]:


house_new_data1


# In[62]:


# after IQR using checking the skewness
for i in ['floor_area_sqm','resale_price']:
    plot(house_new_data1,i)


# In[63]:


plt.figure(figsize= (15,6))
sns.barplot(y = 'flat_model', x= 'resale_price',data = house )


# In[64]:


plt.figure(figsize= (15,6))
sns.barplot(y = 'town', x= 'resale_price',data = house )


# In[65]:


plt.figure(figsize= (15,6))
sns.barplot(y = 'flat_type', x= 'resale_price',data = house )


# In[66]:


plt.figure(figsize=(15,6))
sns.scatterplot(x = 'resale_price' ,y = 'floor_area_sqm',data = house)


# In[67]:


plt.figure(figsize = (12,6))
sns.lineplot(x = 'lease_commence_year' , y = 'resale_price', data = house_new_data)


# In[68]:


plt.figure(figsize = (12,6))
sns.lineplot(x = 'year', y = 'resale_price', data = house_new_data)


# In[69]:


plt.figure(figsize = (12,6))
sns.lineplot(x = 'remaining_lease_years' , y = 'resale_price', data = house_new_data)


# In[78]:


feature_cols = ['town', 'flat_type', 'storey_range',
       'floor_area_sqm', 'flat_model', 'resale_price', 'year', 'month_of_year',
       'lease_commence_year', 'remaining_lease_years',
       'remaining_lease_months']
plt.figure(figsize = (10,4))
plt.title('Correlation Matrix')
sns.heatmap(house_new_data[feature_cols].corr(),cmap="Reds", annot = True)


# In[94]:


# Converting the block column to int
house_new_data['block'] = house_new_data['block'].astype(str)
house_new_data['block'] = house_new_data['block'].apply(lambda x: ''.join(char for char in x if char in '0123456789'))


# In[103]:


house_new_data.dtypes


# ## Train Test Split

# In[108]:


house_2 = house_new_data1[['town', 'flat_type', 'storey_range',
       'floor_area_sqm', 'flat_model', 'resale_price', 'year', 'month_of_year',
       'lease_commence_year', 'remaining_lease_years',
       'remaining_lease_months']]


# In[107]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model   import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle


# In[109]:


house_2.columns


# In[110]:


x = house_2[['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model',
        'year', 'month_of_year', 'lease_commence_year',
       'remaining_lease_years', 'remaining_lease_months']]
y = house_2[['resale_price']]


# In[111]:


#standardize the features of a dataset
encoder = StandardScaler()
encoder.fit_transform(x)


# In[112]:


#check the accuracy of training and testing using metrics RandomForestRegressor


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)
x_train.shape,x_test.shape

RFR = RandomForestRegressor()

RFR= RandomForestRegressor(n_estimators= 50 ,random_state = 0)

# fitting the model: 

RFR.fit(x_train,y_train)

y_pred_train = RFR.predict(x_train)
y_pred_test = RFR.predict(x_test)

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test,y_pred_test)

r2_train,r2_test


# In[ ]:


# GridsearchCV is a cross validation function
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)

param = {'max_depth'        : [20],
              'min_samples_split': [ 5, ],
              'min_samples_leaf' : [ 2, ],
              'max_features'     : ['log2']}
grid_searchcv = GridSearchCV(RandomForestRegressor(),param_grid = param,  cv = 5)
grid_searchcv.fit(x_train, y_train)


# In[ ]:


grid_searchcv.best_score_


# In[113]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)
x_train.shape,x_test.shape

RFR = RandomForestRegressor()

Hyper_model= RandomForestRegressor(max_depth= 20 ,max_features='log2' ,min_samples_leaf=2, min_samples_split=5)

# fitting the model: 

Hyper_model.fit(x_train,y_train)

y_pred_train = Hyper_model.predict(x_train)
y_pred_test = Hyper_model.predict(x_test)

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test,y_pred_test)

r2_train,r2_test


# In[114]:


# predict the selling price with hypertuning parameters and calculate the accuracy using metrics

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)
x_train.shape,x_test.shape

RFR = RandomForestRegressor()

Hyper_model= RandomForestRegressor(max_depth= 20 ,max_features='log2' ,min_samples_leaf=2, min_samples_split=5)

# fitting the model: 

Hyper_model.fit(x_train,y_train)

y_pred_train = Hyper_model.predict(x_train)
y_pred_test = Hyper_model.predict(x_test)                      
print('Mean Squared Error:' ,mean_squared_error(y_test,y_pred_test))
print('Mean Absolute Error:',mean_absolute_error(y_test,y_pred_test))
print('Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test,y_pred_test)))
print(r2_score(y_test,y_pred_test))


# In[115]:


# manually passed the user input and predict the selling price

user_data = np.array([[0,1,3,3.785069,5,2017,1,1979,61.000000,4.000000]])
y_prediction = Hyper_model.predict(user_data)
y_prediction[0]


# In[116]:


user_data = np.array([[4,3,2,4.785069,4,2023,3,1989,69.000000,4.000000]])

y_prediction = Hyper_model.predict(user_data)
y_prediction[0]


# In[117]:


# using Inverse Log Transformation to convert the value to original re sale price of the data (exp)
np.exp(y_prediction[0])


# In[ ]:


# save the regression model by using pickle

with open("C:\\Users\\DELL\\PycharmProjects\\singaporeresale\\resale_model.pkl", 'wb') as f:
    pickle.dump(Hyper_model, f)
     


# In[ ]:


# load the model
with open("C:\\Users\\DELL\\PycharmProjects\\singaporeresale\\resale_model.pkl", 'rb') as f:
    model = pickle.load(f)

user_data = np.array([[4, 3, 2, 4.785069, 4, 2023, 3, 1989, 69.000000, 4.000000]])
prediction = model.predict(user_data)
predicted_price = prediction[0]
predicted_price
np.exp(predicted_price)

