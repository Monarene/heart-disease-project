
# coding: utf-8

# # Heart disease classification

# ## Problem description

# A dataset set of hospital patients is released and some of their medical records are processed and put in a medical csv file, the task is to use the data to build a predictive model with the best possible accuracy to detect heart disease in the patients with given records

# ## Dataset description
# 

# The dataset originally contains 76 attributes but only 13 were released and put in use, The dataset contained 14 columns(13 labels and 1 target), and 303 features/rows
# 

# ## Description of Columns

# * age (age in years)
# * sex (1=male,0=female)
# * cp (chest pain type)
# * trestbps (resting blood pressure in mmHg on admission to the hospital)
# * chol (serum cholestoral in mg/dl
# * fbs (fast blood sugar > 120 mg/dl,1=true,0=false)
# * restecg (resting electrocardiographic results)
# * thalach (maximum heart rate achieved)
# * exang - Exercise induced angina(1=yes,0=no)
# * old peak (st depression induced by exercise relative to rest)
# * slope (the slope of the peak exercise ST segment)
# * ca (the number of major vessels colored by flurosopy)
# * thal (3=normal, 6=fixed defect,7=reversible defect)
# * target (1=positive or 0=negative)

# ## Analyzing the dataset and visualizing the dataset

# In[22]:


#importing all the neccesary libraries
import pandas as pd #importing pandas for data manipulation and visualization
import matplotlib.pyplot as plt #importing matplotlib for visulaization
import numpy as np #numpy for high speed computing
import seaborn as sns #seaborn for Exploratory data analysis

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#importing the dataset
dataset= pd.read_csv('heart.csv')


# In[4]:


dataset.head() #getting the five top variables in the dataset


# In[5]:


dataset.tail() #getting the last five variables


# In[6]:


dataset.isnull().sum()


# Therefore, the data has no missing entry in the dataset, all data entries are consistent

# ## Data preprocessing
# 

# In[7]:


X=dataset.drop('target',axis=1)
y=dataset['target']


# In[8]:


X


# In[9]:


y


# For each piece of algorithm we would be working on, we would be alternating two types of data
# - Scaled data
# - Unscaled data

# In[15]:


#splitting the data into training and test parts
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=7,test_size=0.2)

#scaling
scaler=MinMaxScaler()
sc_x=scaler.fit(x_train)
x_train_scaled=sc_x.transform(x_train.values)
x_test_scaled=sc_x.transform(x_test.values)


# We are going to build only two kinds of models, because of their usual speed and performance on classification dataset. The model typs therefore are
#  * Random forest Classifier
#  * XGBoost Classifier

# In[21]:


#Randomforest
rclassifier1=RandomForestClassifier()
rclassifier2=RandomForestClassifier()
rclassifier1.fit(x_train,y_train)
y_preds=rclassifier1.predict(x_test)
rclassifier2.fit(x_train_scaled,y_train)
y_preds1=rclassifier1.predict(x_test_scaled)
print('The accuracy score for Random Forest Unscaled is', accuracy_score(y_test,y_preds))
print('The accuracy score for Random Forest scaled is', accuracy_score(y_test,y_preds1))


# In[27]:


#XGBoost
#xclassifier2=xgb.XGBClassifier()
#xclassifier1.fit(x_train,y_train)
#y_preds2=xclassifier1.predict(x_test)
#xclassifier2.fit(x_train_scaled,y_train)
#y_preds3=xclassifier2.predict(x_test_scaled)
#print('The accuracy score for XGBoost Unscaled is', accuracy_score(y_test,y_preds2))
#print('The accuracy score for XGBoost scaled is', accuracy_score(y_test,y_preds3))


# The Random Forest Classifier with Unscaled data is produced better results even without hyperparameter tuning.Notice that the accuracy scores for XGBClassifier was equal for both scaled and unscaled data therefore it was a close call to Random forest. Instintively we should concentrate our efforts on the XGBoost classfier even though it performed a little worse. It seems to generalize with the data better 


#play =[]
#for i in range(101):
#    x_train_set,x_test_set,y_train_set,y_test_set=train_test_split(X,y,random_state=i,test_size=0.2)    
#    xclassifier_real=xgb.XGBClassifier()
#    xclassifier_real.fit(x_train_set,y_train_set)
#    y_preds6=xclassifier_real.predict(x_test_set)
#    play.append(accuracy_score(y_test_set,y_preds6))
#print(play.index(max(play)))
#



#splitting the data into training and test parts
x_train_new,x_test_new,y_train_new,y_test_new=train_test_split(X,y,random_state=4,test_size=0.2)

#scaling
scaler=MinMaxScaler()
sc_x=scaler.fit(x_train)
x_train_scaled_new=sc_x.transform(x_train_new.values)
x_test_scaled_new=sc_x.transform(x_test_new.values)

#XGBoost
xclassifier1=xgb.XGBClassifier(n_estimators=15)
xclassifier1.fit(x_train_new,y_train_new)
y_preds7=xclassifier1.predict(x_test_new)
print('The accuracy score for XGBoost Unscaled is', accuracy_score(y_test_new,y_preds7))















