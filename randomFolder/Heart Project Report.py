#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import numpy as np
import pandas as pd
import plotly as plot
import plotly.express as px
import plotly.graph_objs as go
import cufflinks as cf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot


# In[2]:


pyo.init_notebook_mode(connected=True)
cf.go_offline()


# In[3]:


heart=pd.read_csv(r'heart.csv')


# In[4]:


heart


# In[5]:


info = ["age","1:male, 0:female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptotatic","resting blood pressure","serum cholestrol in mg/dl","fasting blood sugar > 120mg/dl","resting cardiographic results (values 0,1,2)", "maximum heart rate achieved","exercise induced angina","ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy"," 3= normal, 6= fixed defect, 7= reversable defect","having heart disease or not, 1: having, 0: not having"]



for i in range(len(info)):
    print(heart.columns[i]+":\t\t\t"+info[i])


# In[6]:


heart['target']


# In[7]:


heart.groupby('target').size()


# In[8]:


heart.groupby('target').sum()


# In[9]:


heart.shape


# In[10]:


heart.size


# In[11]:


heart.describe()


# In[12]:


heart.info()


# In[13]:


heart['target'].unique()


# In[14]:


#visualization


# In[15]:


heart.hist(figsize=(14,14))
plt.show()


# In[16]:


plt.bar(x=heart['sex'],height=heart['age'])
plt.show()


# In[17]:


sns.barplot(x="fbs",y="target",data=heart)
plt.show()


# In[18]:


sns.barplot(x=heart['sex'],y=heart['age'],hue=heart['target'])


# In[19]:


sns.barplot(heart["cp"],heart['target'])


# In[20]:


sns.barplot(heart["sex"],heart['target'])


# In[21]:


px.bar(heart,heart['sex'],heart['target'])


# In[22]:


sns.distplot(heart["thal"])


# In[23]:


sns.distplot(heart["chol"])


# In[24]:


sns.pairplot(heart,hue='target')


# In[25]:


numeric_columns=['trestbps','chol','thalach','age','oldpeak']


# In[26]:


sns.pairplot(heart[numeric_columns])


# In[27]:


heart['target']


# In[28]:


y=heart["target"]
sns.countplot(y)
target_temp=heart.target.value_counts()
print(target_temp)


# In[29]:


#create a corelation heartmap


# In[30]:


sns.heatmap(heart[numeric_columns].corr(),annot=True,cmap='terrain',linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()


# In[31]:


#create four distplots


# In[32]:


plt.figure(figsize=(12,10))
plt.subplot(221)
sns.distplot(heart[heart['target']==0].age)
plt.title('Age of patients without heart disease')
plt.subplot(222)
sns.distplot(heart[heart['target']==1].age)
plt.title('Age of patients with heart disease')
plt.subplot(223)
sns.distplot(heart[heart['target']==0].thalach)
plt.title('Max heart rate of patients without heart disease')
plt.subplot(224)
sns.distplot(heart[heart['target']==1].thalach)
plt.title('Max heart rate of patients with heart disease')
plt.show()


# In[33]:


plt.figure(figsize=(13,6))
plt.subplot(121)
sns.violinplot(x="target",y="thalach",data=heart,inner=None)
sns.swarmplot(x="target",y="thalach",data=heart,color='w',alpha=0.5)

plt.subplot(122)
sns.swarmplot(x="target",y="thalach",data=heart)
plt.show()


# In[34]:


heart


# In[35]:


#create pairplot and two barplots
plt.figure(figsize=(16,6))
plt.subplot(131)
sns.pointplot(x="sex",y="target", hue='cp',data=heart)
plt.legend(['male=1','feamle=0'])
plt.subplot(132)
sns.barplot(x="exang", y="target", data=heart)
plt.legend(['yes=1','no=0'])
plt.subplot(133)
sns.countplot(x="slope",hue="target",data=heart)
plt.show()


# In[36]:


#Data Preprocessing


# In[37]:


heart['target'].value_counts()


# In[38]:


heart['target'].isnull()


# In[39]:


heart['target'].sum()


# In[40]:


heart['target'].unique()


# In[41]:


heart.isnull().sum()


# In[42]:


#storing in x and y


# In[43]:


x,y=heart.loc[:,:'thal'],heart.loc[:,'target']


# In[44]:


x


# In[45]:


y


# In[46]:


###or x,y=heart.iloc[:,:-1],heart.iloc[:,-1]


# In[47]:


x.shape


# In[48]:


y.shape


# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[50]:


x=heart.drop(['target'],axis=1)


# In[51]:


x


# In[52]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.3,shuffle=True)


# ###### x_test

# In[53]:


y_test


# In[54]:


print("train_set_x shape: "+ str(x_train.shape))
print("train_set_y shape: "+ str(y_train.shape))
print("test_set_x shape: "+ str(x_test.shape))
print("test_set_y shape: "+ str(y_test.shape))


# In[55]:


#model


# In[56]:


#decision Tree Classifier


# In[57]:


Category=['No... i pray you dont get Heart Disease or atleast Corona Virus soon...','Yes you have Heart Disease.....RIP in advance']


# In[58]:


from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[59]:


prediction=dt.predict(x_test)
accuracy_dt=accuracy_score(y_test,prediction)*100


# In[60]:


accuracy_dt


# In[61]:


print("Accuracy on training set : {:.3f}".format(dt.score(x_train,y_train)))
print("Accuracy on test set : {:.3f}".format(dt.score(x_test,y_test)))


# In[62]:


y_test


# In[63]:


prediction


# In[64]:


x_DT=np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
x_DT_prediction=dt.predict(x_DT)


# In[65]:


x_DT_prediction[0]


# In[66]:


print(Category[int(x_DT_prediction[0])])


# In[67]:


x_DT_1=np.array([[57,0,0,120,354,0,1,163,1,0.6,2,0,2]])
x_DT_prediction=dt.predict(x_DT_1)


# In[68]:


x_DT_prediction[0]


# In[69]:


print(Category[int(x_DT_prediction[0])])


# In[70]:


x_DT_2=np.array([[45,1,3,110,264,0,1,132,0,1.2,1,0,3]])
x_DT_prediction=dt.predict(x_DT_2)


# In[71]:


x_DT_prediction[0]


# In[72]:


print(Category[int(x_DT_prediction[0])])


# In[73]:


x_DT_3=np.array([[41,0,1,130,204,0,0,172,0,1.4,2,0,2]])
x_DT_prediction=dt.predict(x_DT_3)


# In[74]:


x_DT_prediction[0]


# In[75]:


print(Category[int(x_DT_prediction[0])])


# In[76]:


x_DT_4=np.array([[57,0,1,130,236,0,0,174,0,0.0,1,1,2]])
x_DT_prediction=dt.predict(x_DT_4)


# In[77]:


x_DT_prediction[0]


# In[78]:


print(Category[int(x_DT_prediction[0])])


# In[79]:


x_DT_5=np.array([[68,1,0,144,193,1,1,141,0,3.4,1,2,3]])
x_DT_prediction=dt.predict(x_DT_5)


# In[80]:


x_DT_prediction[0]


# In[81]:


print(Category[int(x_DT_prediction[0])])


# In[82]:


x_DT_6=np.array([[56,1,1,120,236,0,1,178,0,0.8,2,0,2]])
x_DT_prediction=dt.predict(x_DT_6)


# In[83]:


x_DT_prediction[0]


# In[84]:


print(Category[int(x_DT_prediction[0])])


# In[85]:


#Feature Importance in decision tree


# In[86]:


print("Feature importances:\n{}".format(dt.feature_importances_))


# In[87]:


def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features = 13
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),x)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1,n_features)
plot_feature_importances_diabetes(dt)
plt.savefig('feature_importance')


# In[88]:


#KNN


# In[89]:


sc=StandardScaler().fit(x_train)
x_train_std=sc.transform(x_train)
x_test_std=sc.transform(x_test)


# In[90]:


x_test_std


# In[91]:


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train_std,y_train)


# In[92]:


prediction_knn=knn.predict(x_test_std)
accuracy_knn=accuracy_score(y_test,prediction_knn)*100


# In[93]:


accuracy_knn


# In[94]:


print("Accuracy on training set: {:.3f}".format(knn.score(x_train,y_train)))
print("Accuracy on test set: {:.3f}".format(knn.score(x_test,y_test)))


# In[95]:


k_range=range(1,26)
scores={}
scores_list=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_std,y_train)
    prediction_knn=knn.predict(x_test_std)
    scores[k]=accuracy_score(y_test,prediction_knn)
    scores_list.append(accuracy_score(y_test,prediction_knn))


# In[96]:


scores


# In[97]:


plt.plot(k_range,scores_list)


# In[98]:


px.line(x=k_range,y=scores_list)


# In[99]:


x_knn=np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
x_knn_std=sc.transform(x_knn)
x_knn_prediction=dt.predict(x_knn)


# In[100]:


x_knn_std


# In[101]:


x_knn_prediction[0]


# In[102]:


print(Category[int(x_knn_prediction[0])])


# In[103]:


x_knn_1=np.array([[56,1,1,120,236,0,1,178,0,0.8,2,0,2]])
x_knn_std_1=sc.transform(x_knn_1)
x_knn_prediction=dt.predict(x_knn_1)


# In[104]:


x_knn_std_1


# In[105]:


x_knn_prediction[0]


# In[106]:


print(Category[int(x_knn_prediction[0])])


# In[107]:


x_knn_2=np.array([[68,1,0,144,193,1,1,141,0,3.4,1,2,3]])
x_knn_std_2=sc.transform(x_knn_2)
x_knn_prediction=dt.predict(x_knn_2)


# In[108]:


x_knn_std_2


# In[109]:


x_knn_prediction[0]


# In[110]:


print(Category[int(x_knn_prediction[0])])


# In[111]:


x_knn_3=np.array([[41,0,1,130,204,0,0,172,0,1.4,2,0,2]])
x_knn_std_3=sc.transform(x_knn_3)
x_knn_prediction=dt.predict(x_knn_3)


# In[112]:


x_knn_std_3


# In[113]:


x_knn_prediction[0]


# In[114]:


print(Category[int(x_knn_prediction[0])])


# In[115]:


x_knn_4=np.array([[57,0,0,120,354,0,1,163,1,0.6,2,0,2]])
x_knn_std_4=sc.transform(x_knn_4)
x_knn_prediction=dt.predict(x_knn_4)


# In[116]:


x_knn_std_4


# In[117]:


x_knn_prediction[0]


# In[118]:


print(Category[int(x_knn_prediction[0])])


# In[119]:


x_knn_5=np.array([[57,0,1,130,236,0,0,174,0,0.0,1,1,2]])
x_knn_std_5=sc.transform(x_knn_5)
x_knn_prediction=dt.predict(x_knn_5)


# In[120]:


x_knn_std_5


# In[121]:


x_knn_prediction[0]


# In[122]:


print(Category[int(x_knn_prediction[0])])


# In[123]:


algorithms=['Decision Tree','KNN']
scores=[accuracy_dt,accuracy_knn]


# In[124]:


sns.set(rc={'figure.figsize':(15,7)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# In[ ]:




