#!/usr/bin/env python
# coding: utf-8

# In[178]:


import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 4000)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.max_columns",None)


# # Application Data

# In[2]:


df=pd.read_csv('application_data.csv')
df.head()


# In[3]:


df.shape


# In[4]:


print(df.info('all'))


# In[5]:


df.isnull().sum()


# In[6]:


df.isnull().mean()


# #  Drop Column

# In[7]:


df.describe()


# In[8]:


def Temp(dataframe):
    return round((dataframe.isnull().sum()*100/len(dataframe)).sort_values(ascending=False),3)
                 


# In[9]:


Temp(df)


# In[10]:


print(Temp(df)[Temp(df)>50])
print("Total columns",Temp(df)[Temp(df)>50].count())


# In[11]:


drop_colm=Temp(df)[Temp(df)>50]
drop_colm


# In[12]:


drop_colm.index


# In[13]:


df.drop(columns=drop_colm.index,inplace=True)


# In[14]:


df.shape


# In[15]:


Temp(df)[Temp(df)>40]


# In[16]:


loan=Temp(df)[Temp(df)>40]
loan


# In[17]:


loan.index


# In[18]:


df.drop(columns=loan.index,inplace=True)


# In[19]:


df.shape


# In[20]:


df.head()


# In[ ]:





# # Missing Values

# In[21]:


df.isnull().sum()


# In[22]:


df.info('all')


# #  Outliers from boxplot 
# - Median

# In[23]:


df.AMT_ANNUITY.value_counts()


# In[24]:


sns.boxplot(df.AMT_ANNUITY)
plt.show()


# In[25]:


df.AMT_ANNUITY.median()


# In[26]:


df['AMT_ANNUITY']=df.AMT_ANNUITY.fillna(df.AMT_ANNUITY.median())


# In[27]:


df.AMT_ANNUITY.isnull().sum()


# In[28]:


df.AMT_GOODS_PRICE.value_counts()


# In[29]:


sns.boxplot(df.AMT_GOODS_PRICE)
plt.show()


# In[30]:


df.AMT_GOODS_PRICE.median()


# In[31]:


df['AMT_GOODS_PRICE']=df.AMT_GOODS_PRICE.fillna(df.AMT_GOODS_PRICE.median())


# In[32]:


df.AMT_GOODS_PRICE.isnull().sum()


# In[33]:


df.NAME_TYPE_SUITE.value_counts()


# In[34]:


df.NAME_TYPE_SUITE.isnull().sum()


# In[35]:


df.info('NAME_TYPE_SUITE')


# In[36]:


df['NAME_TYPE_SUITE']=df.NAME_TYPE_SUITE.fillna('Unaccompanied')


# In[37]:


df['NAME_TYPE_SUITE'].isnull().sum()


# In[38]:


df.OCCUPATION_TYPE.value_counts()


# In[39]:


df.info('OCCUPATION_TYPE')


# In[40]:


df.OCCUPATION_TYPE.isnull().sum()


# In[41]:


df['OCCUPATION_TYPE']=df.OCCUPATION_TYPE.fillna('Unknown')


# In[42]:


df.OCCUPATION_TYPE.isnull().sum()


# In[43]:


df.CNT_FAM_MEMBERS.value_counts()


# In[44]:


df.CNT_FAM_MEMBERS.isnull().sum()


# In[45]:


sns.boxplot(df.CNT_FAM_MEMBERS)
plt.show()


# In[46]:


df.CNT_FAM_MEMBERS.median()


# In[47]:


df['CNT_FAM_MEMBERS']=df.CNT_FAM_MEMBERS.fillna(df.CNT_FAM_MEMBERS.median())


# In[48]:


df.CNT_FAM_MEMBERS.isna().sum()


# In[49]:


df.EXT_SOURCE_2.value_counts()


# In[50]:


df.EXT_SOURCE_2.isnull().sum()


# In[51]:


df.EXT_SOURCE_3.value_counts()


# In[52]:


df.EXT_SOURCE_3.isnull().sum()


# In[53]:


plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.title("EXT_SOURCE_2")
sns.boxplot(df.EXT_SOURCE_2)

plt.subplot(2,2,2)
plt.title("EXT_SOURCE_3")
sns.boxplot(df.EXT_SOURCE_3)


# In[54]:


df[['EXT_SOURCE_2','EXT_SOURCE_3']].median()


# In[55]:


df['EXT_SOURCE_2']=df.EXT_SOURCE_2.fillna(df['EXT_SOURCE_2'].median())


# In[56]:


df.EXT_SOURCE_2.isnull().sum()


# In[57]:


df['EXT_SOURCE_3']=df.EXT_SOURCE_2.fillna(df['EXT_SOURCE_3'].median())


# In[58]:


df.EXT_SOURCE_3.isnull().sum()


# In[59]:


df[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']].describe()


# In[60]:


plt.figure(figsize=(20,5))
plt.subplot(2,2,1)
plt.title("OBS_30")
sns.boxplot(df.OBS_30_CNT_SOCIAL_CIRCLE)

plt.subplot(2,2,2)
plt.title("DEF_30")
sns.boxplot(df.DEF_30_CNT_SOCIAL_CIRCLE)

plt.subplot(2,2,3)
plt.title("OBS_60")
sns.boxplot(df.OBS_60_CNT_SOCIAL_CIRCLE)

plt.subplot(2,2,4)
plt.title("DEF_60")
sns.boxplot(df.DEF_60_CNT_SOCIAL_CIRCLE)


# In[61]:


df[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']].median()


# In[62]:


CNT_SOCIAL_CIRCLE=['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']


# In[63]:


df.fillna(df[CNT_SOCIAL_CIRCLE].median(),inplace=True)


# In[64]:


df[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']].isnull().sum()


# In[65]:


df.DAYS_LAST_PHONE_CHANGE.value_counts()


# In[66]:


df.DAYS_LAST_PHONE_CHANGE.isnull().sum()


# In[67]:


sns.boxplot(df.DAYS_LAST_PHONE_CHANGE)
plt.show()


# In[68]:


df.DAYS_LAST_PHONE_CHANGE.median()


# In[69]:


df.fillna(df.DAYS_LAST_PHONE_CHANGE.median(),inplace=True)


# In[70]:


df.DAYS_LAST_PHONE_CHANGE.isnull().sum()


# In[71]:


df[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].describe()


# In[72]:


plt.figure(figsize=(20,20))
plt.subplot(3,2,1)
plt.title("BUREAU_HOUR")
sns.boxplot(df.AMT_REQ_CREDIT_BUREAU_HOUR)

plt.subplot(3,2,2)
plt.title("BUREAU_DAY")
sns.boxplot(df.AMT_REQ_CREDIT_BUREAU_DAY)

plt.subplot(3,2,3)
plt.title("BUREAU_WEEK")
sns.boxplot(df.AMT_REQ_CREDIT_BUREAU_WEEK)

plt.subplot(3,2,4)
plt.title("BUREAU_MON")
sns.boxplot(df.AMT_REQ_CREDIT_BUREAU_MON)

plt.subplot(3,2,5)
plt.title("BUREAU_QRT")
sns.boxplot(df.AMT_REQ_CREDIT_BUREAU_QRT)

plt.subplot(3,2,6)
plt.title("BUREAU_YEAR")
sns.boxplot(df.AMT_REQ_CREDIT_BUREAU_YEAR)


# # Replacing Null Values

# In[73]:


AMT_REQ_CREDIT=['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
                'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']


# In[74]:


df[AMT_REQ_CREDIT].median()


# In[75]:


df.fillna(df[AMT_REQ_CREDIT].median(),inplace=True)


# In[76]:


df[AMT_REQ_CREDIT].isnull().sum()


# In[77]:


df.isnull().sum()


# In[78]:


df.shape


# # Data Cleansing and Manipulation

# In[79]:


df.describe()


# Collecting columns starting with 'DAYS' in the list 'day_column'

# In[80]:


day_column=[i for i in df if i.startswith('DAYS')]
day_column


# In[81]:


df[day_column]=abs(df[day_column])


# In[82]:


print(df['DAYS_BIRTH'].unique())
print(df['DAYS_EMPLOYED'].unique())
print(df['DAYS_REGISTRATION'].unique())
print(df['DAYS_ID_PUBLISH'].unique())
print(df['DAYS_LAST_PHONE_CHANGE'].unique())


# In[83]:


df[["DAYS_BIRTH","DAYS_EMPLOYED",
   "DAYS_REGISTRATION","DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE"]]=abs(df[["DAYS_BIRTH","DAYS_EMPLOYED",
    "DAYS_REGISTRATION","DAYS_ID_PUBLISH","DAYS_LAST_PHONE_CHANGE"]])


# In[84]:


df[["DAYS_BIRTH","DAYS_EMPLOYED",
   "DAYS_REGISTRATION","DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE"]].describe()


# In[ ]:





# #  Binning of continuous Variables
# 

# In[85]:


df['AGE']=round(df['DAYS_BIRTH']/365,2)


# In[86]:


df['AGE_RANGE']=pd.cut(df['AGE'],bins=[0,20,25,30,35,40,45,50,55,60,65,70],labels=["0-20","20-25","25-30","30-35","35-40","40-45","45-50","50-55","55-60","60-65",'above65'])


# In[87]:


df[['AGE_RANGE','AGE']]


# In[88]:


sns.countplot(df['AGE_RANGE'])
plt.show()


# In[89]:


df['EMPLOYED_DAYS']=round(df['DAYS_EMPLOYED']/365,2)


# In[90]:


df['EMPLOYED_DAYS_RANGE']=pd.cut(df['EMPLOYED_DAYS'],bins=[0,5,10,15,20,25,30,35,40,45,50,55],labels=['0-5','5-10','10-15','15-20','20-25','25-30','30-35','35-40','40-45','45-50','above 50'])


# In[91]:


df[['EMPLOYED_DAYS_RANGE','EMPLOYED_DAYS']]


# In[92]:


df['EMPLOYED_DAYS']=df.EMPLOYED_DAYS.replace(df.EMPLOYED_DAYS.max(),np.NAN)


# In[93]:


sns.countplot(df['EMPLOYED_DAYS_RANGE'])
plt.show()


# In[94]:


df['ATM_INCOME']=round(df['AMT_INCOME_TOTAL']/100000,2)


# In[95]:


df['ATM_INCOME_RANGE']=pd.cut(df['ATM_INCOME'],bins=[0,1,2,3,4,5,6,7,8,9,10,100],labels=['0-1L','1-2L','2-3L','3-4L','4-5L','5-6L','6-7L','7-8L','8-9L','9-10L','above 10L'])


# In[96]:


df[['ATM_INCOME_RANGE','ATM_INCOME']]


# In[97]:


sns.countplot(df['ATM_INCOME_RANGE'])
plt.show()


# In[98]:


df['ATM_CREDIT_1']=round(df['AMT_CREDIT']/100000,2)


# In[99]:


df['ATM_CREDIT_1_RANGE']=pd.cut(df['ATM_CREDIT_1'],bins=[0,5,10,15,20,25,30,35,40,45],labels=['0-5L','5-10L','10-15L','15-20L','20-25L','25-30L','30-35L','35-40L','above 40L'])


# In[100]:


df[['ATM_CREDIT_1_RANGE','AMT_CREDIT']]


# In[101]:


sns.countplot(df['ATM_CREDIT_1_RANGE'])
plt.show()


# In[102]:


df['Credit_Ratio']=round(df.AMT_CREDIT/df.AMT_INCOME_TOTAL,2)
df['Credit_Ratio']


# In[103]:


sns.countplot(df['Credit_Ratio'])
plt.show()


# # TARGET 

# In[104]:


df.TARGET.value_counts(normalize=True)


# In[105]:


df.TARGET.value_counts(normalize=True).plot.pie()
plt.show()


# In[106]:


Target_payment_difficulty=df[df.TARGET==1]
Target_all_other=df[df.TARGET==0]


# In[107]:


round(len(Target_all_other)/len(Target_payment_difficulty),2)


# In[108]:


plt.figure(figsize=(15,7))
plt.subplot(121)
sns.countplot(x="TARGET",hue="AGE_RANGE",data=Target_all_other)
plt.subplot(122)
sns.countplot(x="TARGET",hue="AGE_RANGE",data=Target_payment_difficulty)
plt.show()


# In[109]:


plt.figure(figsize=(15,7))
plt.subplot(121)
sns.countplot(x="TARGET",hue="CODE_GENDER",data=Target_all_other)
plt.title("Target_all_other")
plt.subplot(122)
sns.countplot(x="TARGET",hue="CODE_GENDER",data=Target_payment_difficulty)
plt.title("Target_payment_difficulty")
plt.show()


# In[110]:


Target_all_other.CODE_GENDER.value_counts()


# In[111]:


plt.figure(figsize=(15,7))
plt.subplot(121)
sns.countplot(x="TARGET",hue="CNT_FAM_MEMBERS",data=Target_all_other)
plt.title("Target_all_other")
plt.subplot(122)
sns.countplot(x="TARGET",hue="CNT_FAM_MEMBERS",data=Target_payment_difficulty)
plt.title("Target_payment_difficulty")
plt.show()


# # Univariate Analysis
# - Categorical Data and Numerical Data

# In[112]:


df.head()


# In[113]:


Numerical=['Credit_Ratio','ATM_CREDIT_1','ATM_INCOME','EMPLOYED_DAYS','AGE',
                          'AMT_GOODS_PRICE','AMT_ANNUITY','CNT_FAM_MEMBERS']


# In[114]:


def Uni_Numerical(dataframe, column):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,3,1)
    sns.boxplot(data=dataframe, x=column, orient='v').set(title='BoxPlot')
    
    plt.subplot(1,3,2)
    sns.distplot(dataframe[column].dropna()).set(title='Distplot')
    plt.show()


# In[115]:


for i in Numerical:
    Uni_Numerical(df,i)


# In[ ]:





# In[116]:


Categorical_col=df.select_dtypes(include='object').columns
Categorical_col


# In[117]:


Categorical=['NAME_CONTRACT_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY',"NAME_TYPE_SUITE",
             'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
             'NAME_HOUSING_TYPE','OCCUPATION_TYPE','AGE_RANGE','EMPLOYED_DAYS_RANGE',
             'ATM_CREDIT_1_RANGE','ATM_INCOME_RANGE']


# In[118]:


def Uni_Categorical(dataframe, column):
    sns.set(style='darkgrid')
    plt.figure(figsize=[15,5])
    dataframe[column].value_counts().plot.barh(width=0.8)
    plt.show()


# In[119]:


for i in Categorical:
    Uni_Categorical(df,i) 


# # Bivariate Analysis 
# # Target

# In[120]:


sns.pairplot(data=df,vars=['AMT_ANNUITY','AMT_GOODS_PRICE','ATM_CREDIT_1','ATM_INCOME'])
plt.show()


# In[121]:


sns.scatterplot(x=Target_payment_difficulty.AMT_ANNUITY, 
                 y=Target_payment_difficulty.AMT_GOODS_PRICE,
                data=Target_payment_difficulty, hue='TARGET')


# In[122]:


plt.figure(figsize=[10,6])
sns.set(style='darkgrid')
sns.barplot(x=df.CODE_GENDER, y=df.AGE)
plt.show()


# In[123]:


sns.pairplot(df,vars=['ATM_CREDIT_1','ATM_INCOME','OCCUPATION_TYPE'],
            diag_kind='hist', hue='TARGET', plot_kws={'alpha':0.6,'s':80,'edgecolor':'b'},size=4)
plt.show()


# In[124]:


plt.figure(figsize=(25,5))
sns.heatmap(Target_payment_difficulty.corr(),cmap="Blues")
plt.title("payment_difficulty")
plt.show()


# In[125]:


df3=Target_payment_difficulty[['CNT_CHILDREN','AMT_ANNUITY',
                               'AMT_GOODS_PRICE','AGE','EMPLOYED_DAYS','ATM_INCOME',
                              'ATM_CREDIT_1','CNT_FAM_MEMBERS','FLAG_OWN_REALTY',
                               'NAME_TYPE_SUITE',
                               'NAME_INCOME_TYPE',
                               'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                               'OCCUPATION_TYPE','NAME_HOUSING_TYPE']]
plt.figure(figsize=(25,10))
sns.heatmap(df3.corr(method='pearson'),annot=True)
plt.title("payment_difficulty")
plt.show()


# In[ ]:





# In[ ]:





# # Previous application
# - Data Structure

# In[126]:


df1=pd.read_csv("previous_application.csv")
df1.head()


# In[127]:


df1.shape


# In[128]:


print(df1.info('all'))


# In[129]:


df1.isnull().sum()


# In[130]:


df1.isnull().mean()


# In[131]:


df1.describe()


# # Dropping Column

# In[132]:


Temp(df1)


# In[133]:


drop_colm1=Temp(df1)[Temp(df1)>50]
drop_colm1


# In[134]:


drop_colm1.index


# In[135]:


df1.drop(columns=drop_colm1.index,inplace=True)


# In[136]:


Temp(df1)


# In[137]:


S=Temp(df1)[Temp(df1)>40]
S


# - Filling Null values

# In[138]:


df1.NAME_TYPE_SUITE.value_counts()


# In[139]:


df1.NAME_TYPE_SUITE.isnull().sum()


# In[140]:


df1['NAME_TYPE_SUITE']=df1.NAME_TYPE_SUITE.fillna('Unaccompanied')


# In[141]:


df1.AMT_GOODS_PRICE.describe()


# In[142]:


sns.boxplot(df1.AMT_GOODS_PRICE)
plt.show()


# In[143]:


df1.AMT_GOODS_PRICE.median()


# In[144]:


df1["AMT_GOODS_PRICE"]=df1.AMT_GOODS_PRICE.fillna(df1['AMT_GOODS_PRICE']==df1['AMT_CREDIT'])  


# In[145]:


df1.AMT_GOODS_PRICE.isnull().sum()


# In[146]:


df1.AMT_CREDIT.isnull().sum()


# In[147]:


df1.AMT_ANNUITY.describe()


# In[148]:


sns.boxplot(df1.AMT_ANNUITY)
plt.show()


# In[149]:


df1.AMT_ANNUITY.median()


# In[150]:


df1['AMT_ANNUITY']=df1.AMT_ANNUITY.fillna(df1.AMT_ANNUITY.median())


# In[151]:


df1.AMT_ANNUITY.isnull().sum()


# In[152]:


df1.CNT_PAYMENT.describe()


# In[153]:


sns.boxplot(df1.CNT_PAYMENT)
plt.show()


# In[154]:


df1.CNT_PAYMENT.median()


# In[155]:


df1['CNT_PAYMENT']=df1.CNT_PAYMENT.fillna(df1.CNT_PAYMENT.median())


# In[156]:


df1.CNT_PAYMENT.isnull().sum()


# In[157]:


df1.PRODUCT_COMBINATION.describe() 


# In[158]:


df1.PRODUCT_COMBINATION.mode()[0]


# In[159]:


df1['PRODUCT_COMBINATION']=df1.PRODUCT_COMBINATION.fillna(df1.PRODUCT_COMBINATION.mode()[0])


# In[160]:


df1.PRODUCT_COMBINATION.value_counts()


# In[161]:


df1.PRODUCT_COMBINATION.value_counts().plot.density()


# In[162]:


df1.NFLAG_INSURED_ON_APPROVAL.value_counts()


# In[163]:


Temp(df1)


# # Binning of continuous Variables

# In[164]:


df1.describe()


# In[165]:


df1['AMT_ANNUITY_AMOUNT']=df1['AMT_ANNUITY']/100000


# In[166]:


df1['AMT_APPLICATION_AMOUNT']=df1['AMT_APPLICATION']/100000


# In[167]:


df1['AMT_CREDIT_AMOUNT']=df1['AMT_CREDIT']/100000


# In[168]:


df1[['AMT_ANNUITY_AMOUNT','AMT_APPLICATION_AMOUNT','AMT_CREDIT_AMOUNT']].describe()


# - Applying abs() function to columns starting with 'DAYS' to convert the negative values to positive.

# In[169]:


df1[['DAYS_DECISION','DAYS_FIRST_DRAWING',
     'DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION',
     'DAYS_LAST_DUE','DAYS_TERMINATION']]=abs(df1[['DAYS_DECISION','DAYS_FIRST_DRAWING',
     'DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION',
     'DAYS_LAST_DUE','DAYS_TERMINATION']])


# In[170]:


df1[['DAYS_DECISION','DAYS_FIRST_DRAWING',
     'DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION',
     'DAYS_LAST_DUE','DAYS_TERMINATION']]=round(df1[['DAYS_DECISION','DAYS_FIRST_DRAWING',
     'DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION',
     'DAYS_LAST_DUE','DAYS_TERMINATION']]/365,2)


# In[171]:


df1[['DAYS_DECISION','DAYS_FIRST_DRAWING',
     'DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION',
     'DAYS_LAST_DUE','DAYS_TERMINATION']].describe()


# In[172]:


df1['AMT_CREDIT_AMOUNT_RANGE']=pd.cut(df['ATM_INCOME'],
                                      bins=[0,1,2,3,4,5,6,7,8,9,10,100],
                                      labels=['0-1L','1-2L','2-3L','3-4L',
                                      '4-5L','5-6L','6-7L','7-8L','8-9L','9-10L','above 10L'])
df1['AMT_APPLICATION_AMOUNT_RANGE']=pd.cut(df['ATM_INCOME'],
                                          bins=[0,1,2,3,4,5,6,7,8,9,10,100],
                                          labels=['0-1L','1-2L','2-3L','3-4L',
                                          '4-5L','5-6L','6-7L','7-8L','8-9L','9-10L','above 10L'])
                                                                     


# In[173]:


df1.head(4)


# # Univariate Analysis

# In[174]:


df1.head()


# In[175]:


Categorical_Data=['NAME_CONTRACT_TYPE',
                  'NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE',
                 'NAME_TYPE_SUITE','NAME_CLIENT_TYPE',
                  'AMT_CREDIT_AMOUNT_RANGE','AMT_APPLICATION_AMOUNT_RANGE']


# In[176]:


for i in Categorical_Data:
    Uni_Categorical(df1,i)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




