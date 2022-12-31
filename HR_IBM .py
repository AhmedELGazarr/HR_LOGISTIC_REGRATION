#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
import matplotlib.pyplot as plt
sns.set_theme()
from sklearn.model_selection import train_test_split
df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

#!pip install sweetviz
import sweetviz as sv
#!pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()


# In[6]:


report_alcohol = sv.analyze(df)


# In[7]:


report_alcohol.show_notebook()


# In[9]:


from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()


# In[10]:


from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
a = AV.AutoViz('Ecommerce Purchases.csv')


# # BREPROCESSING & CLEANING DATA..

# In[10]:


df = AV.AutoViz('"WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[145]:


#null values
pd.isnull(df).sum()


# In[147]:


#data info
df.info()


# In[11]:


df.head()


# In[12]:


# name of my columns
df.columns


# In[13]:


df.duplicated().sum()


# In[14]:


df.shape


# In[15]:


# my nunique values
for column in df.columns:
        print("{}:{}".format(column,df.columns.nunique()))


# In[16]:


for col in df.columns:
    print("{}:{}".format(col,df[col].nunique()))


# In[17]:


for c in df.columns:
    if df[c].dtype== object:
        print("{}:{}".format(c,df[c].nunique()))


# In[18]:


for c in df.columns:
    if df[c].dtype != object:
        print("{}:{}".format(c,df[c].nunique()))        


# In[19]:


df.describe()


# In[20]:


# drop the columns
df.drop(['Over18','EmployeeCount','EmployeeNumber','StandardHours'],axis=1, inplace=True)


# In[21]:


# the corr map
plt.figure(figsize=(15,15))
sns.heatmap(df.corr().round(3),annot=True,);


# In[22]:


#the number of Attrition
sns.countplot(x=df.Attrition);


# # Scatter

# In[23]:


sns.pairplot(df);


# # Display The Relationship Around The Entity.

# In[24]:


for i in df.columns :
    sns.histplot(x=df[i])
    plt.show();


# In[25]:


df.groupby(['Department','Gender'])['Attrition'].value_counts()


# In[26]:


df[df.Age==30]['Attrition'].value_counts()


# In[27]:


df.groupby(['JobRole'])['Attrition'].value_counts(normalize=True).round(2)


# In[28]:


df.groupby(['JobRole'])['MonthlyIncome'].mean().round(2)


# #   machine Learing Model

# 1.	Change categorical data to numerical 
# 2.	Transform the data to (0,1)
# 3.	Split test &predict
# 4.	Standard the data 
# 5.	Clustering test 
# 6.	Evaluate The Test  
# 

# In[31]:


x= df.loc[:,df.columns!="Attrition"]
y=df.Attrition


# In[32]:


x


# In[124]:


#import the tools
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression


# In[125]:


LabelEncoder = LabelEncoder()
y =LabelEncoder.fit_transform(y)
x=pd.get_dummies(x, drop_first=True)


# In[128]:


x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3,random_state=1, stratify=y)


# In[129]:


scaler=StandardScaler()
x_train=pd.DataFrame(scaler.fit_transform(x_train), columns= x_train.columns)
x_test= pd.DataFrame(scaler.transform(x_test),columns= x_train.columns)


# In[130]:


x_train.head()


# In[131]:


#creat my model and put num to stop on it 
model= LogisticRegression(max_iter=5000) #model building


# In[132]:


#apply my function and test with train
model.fit(x_train,y_train) #model training


# In[133]:


#model prediction 
y_pre=model.predict(x_test)


# In[134]:


y_pre


# 
# # Model Evalution  by LogisticRegression

# In[135]:


#importing the evaluation tool
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pre,y_test)


# In[136]:


#import the tool that iam going to use for my evalution
from sklearn.metrics import classification_report,confusion_matrix
# evaltion by report 
# if i changed LogisticRegression the precision will change
print(classification_report(y_pre,y_test))


# #  Building additionl Model by using SVC .

# In[138]:


#import my model from the liberary
# LogisticRegression this what detrime if i true or not 
from sklearn.svm import SVC


# In[139]:


model_2= SVC() #model building
#apply my function and test with train
model_2.fit(x_train,y_train)#model training
#model prediction 
y_pre_2=model.predict(x_test)


# # Model Evalution  by SVC .

# In[142]:


#import the tool that iam going to use for my evalution
from sklearn.metrics import classification_report
# evaltion by report 
# if i changed SVC the precision will change
print(classification_report(y_pre_2,y_test))


# # AHMED El-GAZAR
