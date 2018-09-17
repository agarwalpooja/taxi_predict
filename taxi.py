
# coding: utf-8

# In[1]:


import pandas as pd
import datetime as dt
import numpy as np 
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


print("Loading Training and Testing Data =====>")
training_data = pd.read_csv(r'C:\Users\ASUS\Documents\jbg_ml\taxi\train\train.csv')
testing_data = pd.read_csv(r'C:\Users\ASUS\Documents\jbg_ml\taxi\test\test.csv')
print("<===== Training and Testing Data Loading finished")
sample=pd.read_csv(r'C:\Users\ASUS\Documents\jbg_ml\taxi\sample_submission\sample_submission.csv')


# In[3]:


training_data.describe()


# In[4]:


training_data.info()


# In[5]:


training_data.head()


# In[6]:


training_data.id.nunique()


# In[7]:


training_data.groupby('vendor_id').vendor_id.count()


# In[8]:


np.max(training_data.passenger_count)


# In[9]:


testing_data.info()


# In[10]:


testing_data.head()


# In[11]:


sample.info()


# In[12]:


def convert_datetime(s):
    if type(s)==str:
        return dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    else:
        return s
    
def get_hour(d):
    return d.hour

def get_weekday(d):
    weekday = d.isoweekday()
    return weekday

# def get_geohash(row):
#     return geohash.encode(row['pickup_latitude'], row['pickup_longitude'], precision=6)

def get_distance(lat1, long1, lat2, long2):
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(long1)
    
    lat2 = radians(lat2)
    lon2 = radians(long2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    
    #in km
    return distance

def distance(row):
    return get_distance(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude'])


# In[13]:


training_data.pickup_datetime = training_data.pickup_datetime.apply(convert_datetime)
training_data['hour'] = training_data.pickup_datetime.apply(get_hour)
training_data['weekday'] = training_data.pickup_datetime.apply(get_weekday)
training_data['month'] = training_data.pickup_datetime.dt.month
training_data['Eucl_distance'] = training_data.apply(distance, axis=1)


# In[14]:


training_data.describe()


# In[15]:


print("Number of ids in the train dataset: ", len(training_data["id"]))
print("Number of unique ids in the train dataset: ", len(pd.unique(training_data["id"])), "\n")

print("Number of ids in the test dataset: ", len(testing_data["id"]))
print("Number of unique ids in the test dataset: ", len(pd.unique(testing_data["id"])), "\n")

print("Number of common ids(if any) between the train and test datasets: ", len(set(training_data["id"].values).intersection(set(testing_data["id"].values))))


# In[16]:


print("Number of vendor_ids in the train dataset: ", len(training_data["vendor_id"]))
print("Number of unique vendor_ids in the train dataset: ", len(pd.unique(training_data["vendor_id"])), "\n")

print("Number of vendor_ids in the test dataset: ", len(testing_data["vendor_id"]))
print("Number of unique vendor_ids in the test dataset: ", len(pd.unique(testing_data["vendor_id"])), "\n")


# In[17]:


sns.countplot(x="vendor_id",data=training_data)


# In[18]:


sns.countplot(x="vendor_id",data=testing_data)


# In[19]:


testing_data.pickup_datetime = testing_data.pickup_datetime.apply(convert_datetime)
training_data.dropoff_datetime = training_data.dropoff_datetime.apply(convert_datetime)


# In[20]:


# #trip_duration represents the difference between the dropoff_datetime and the pickup_datetime in the
# #train dataset
training_data["trip_duration"].describe()


# In[21]:


(training_data["dropoff_datetime"] - training_data["pickup_datetime"]).describe()


# In[22]:


plt.figure(figsize=(10,10))
plt.scatter(range(len(training_data["trip_duration"]/3600)), np.sort(training_data["trip_duration"]/3600))
plt.xlabel('index')
plt.ylabel('trip_duration in seconds')
plt.show()


# In[23]:


training_data = training_data[training_data["trip_duration"] < 500000]


# In[24]:


(training_data["dropoff_datetime"] - training_data["pickup_datetime"]).describe()


# In[25]:


plt.figure(figsize=(10,10))
plt.scatter(range(len(training_data["trip_duration"]/3600)), np.sort(training_data["trip_duration"]/3600))
plt.xlabel('index')
plt.ylabel('trip_duration in seconds')
plt.show()


# In[26]:


sns.countplot(x="store_and_fwd_flag", data=training_data)


# In[27]:


len(training_data[training_data["store_and_fwd_flag"] == "N"])*100.0/(training_data.count()[0])


# In[28]:


set(training_data[training_data["store_and_fwd_flag"] == "Y"]["vendor_id"])


# In[29]:


plt.figure(figsize=(10,10))
plt.scatter(range(len(training_data["Eucl_distance"])), np.sort(training_data["Eucl_distance"]))
plt.xlabel('index')
plt.ylabel('Eucl_distance in meters')
plt.show()


# In[30]:


training_data = training_data[training_data["Eucl_distance"] < 600]


# In[31]:


plt.figure(figsize=(10,10))
plt.scatter(range(len(training_data["Eucl_distance"])), np.sort(training_data["Eucl_distance"]))
plt.xlabel('index')
plt.ylabel('Eucl_distance in meters')
plt.show()


# In[32]:


def count_elements(array):
    return len(pd.unique(array))


# In[33]:


plt.figure(figsize=(10, 12))
pivot_2 = training_data.pivot_table(index='hour' , columns='weekday', values='id', aggfunc=count_elements)
pivot_2.sort_index(level=0, ascending=False, inplace=True)
ax2 = sns.heatmap(pivot_2)
ax2.set_title('# of Rides per Pickup Hour and Weekday')
plt.show()


# In[34]:


plt.figure(figsize=(10, 12))
pivot_1 = training_data.pivot_table(index='hour' , columns='weekday', values='trip_duration', aggfunc=np.mean)
pivot_1.sort_index(level=0, ascending=False, inplace=True)
ax3 = sns.heatmap(pivot_1)
ax3.set_title('Trip Duration [seconds] per Pickup Hour and Weekday')
plt.show()


# In[35]:


tab = training_data.pivot_table(index=['weekday', 'hour'] , values=['trip_duration', 'Eucl_distance'], aggfunc=np.sum)
tab['avg_velocity'] = tab['Eucl_distance'] / (tab['trip_duration']/3600)
tab = tab.reset_index()
tab.avg_velocity = tab.avg_velocity.astype(int)
plt.figure(figsize=(10, 12))
tab = tab.pivot('hour', 'weekday', 'avg_velocity')
tab.sort_index(level=0, ascending=False, inplace=True)
ax4 = sns.heatmap(tab, annot=True, fmt='d')
ax4.set_title('Avg Velocity [km/h] per Pickup hour and weekday')
plt.show()


# In[36]:


training_data.trip_duration.describe()


# In[37]:


df_train_agg = training_data.groupby('weekday')['trip_duration'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.weekday.values, df_train_agg.trip_duration.values)
plt.show()


# In[38]:


training_data.groupby('weekday')['trip_duration'].describe()


# In[39]:


df_train_agg = training_data.groupby('weekday')['trip_duration'].aggregate(np.mean).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.weekday.values, df_train_agg.trip_duration.values)
plt.show()


# In[40]:


df_train_agg = training_data.groupby('hour')['trip_duration'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.hour.values, df_train_agg.trip_duration.values)
plt.show()


# In[41]:


training_data.groupby('hour')['trip_duration'].describe()


# In[42]:


df_train_agg = training_data.groupby('month')['trip_duration'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.month.values, df_train_agg.trip_duration.values)
plt.show()
training_data.groupby('month')['trip_duration'].describe()


# In[43]:


# def rmsle(y_test, y_pred) : 
#     assert len(y_test) == len(y_pred)
#     return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
# kf = KFold(n_splits=3)
# neighbors_array=[]
# training_score=[]
# testing_score=[]
# for n_neighbors in range(1,10,2):
#     knn = KNeighborsRegressor(n_neighbors=n_neighbors)
#     print('\nEvaluating metrics with n_neighbors= ' + str(n_neighbors))
#     neighbors_array.append(n_neighbors)
#     ind_test_score=[]
#     ind_train_score=[]
    
#     for train_index, test_index in kf.split(training_data):
#         train_x = training_data.loc[train_index, ['hour', 'weekday', 'Eucl_distance']]
#         train_y = training_data.loc[train_index, ['trip_duration']]

#         test_x = training_data.loc[test_index, ['hour', 'weekday', 'Eucl_distance']]
#         test_y = training_data.loc[test_index, ['trip_duration']]

#         knn.fit(train_x, train_y)
#         y_pred_test = knn.predict(test_x)
#         y_pred_train = knn.predict(train_x)
        
#         ind_train_score.append(rmsle(train_y, y_pred_train))
#         ind_test_score.append(rmsle(test_y, y_pred_test))
#     training_score.append(np.mean(ind_train_score))
#     testing_score.append(np.mean(ind_test_score))


# In[44]:


# kf = KFold(n_splits=3)
# neighbors_array=[]
# training_score=[]
# testing_score=[]
# for n_neighbors in range(1,10,2):
#     knn = KNeighborsRegressor(n_neighbors=n_neighbors)
#     print('\nEvaluating metrics with n_neighbors= ' + str(n_neighbors))
#     neighbors_array.append(n_neighbors)
#     ind_test_score=[]
#     ind_train_score=[]
    
#     for train_index, test_index in kf.split(training_data):
#         train_x = training_data.loc[train_index, ['hour', 'weekday', 'Eucl_distance','store_and_fwd_flag']]
#         train_y = training_data.loc[train_index, ['trip_duration']]

#         test_x = training_data.loc[test_index, ['hour', 'weekday', 'Eucl_distance','store_and_fwd_flag']]
#         test_y = training_data.loc[test_index, ['trip_duration']]

#         knn.fit(train_x, train_y)
#         y_pred_test = knn.predict(test_x)
#         y_pred_train = knn.predict(train_x)
        
#         ind_train_score.append(rmsle(train_y, y_pred_train))
#         ind_test_score.append(rmsle(test_y, y_pred_test))
#     training_score.append(np.mean(ind_train_score))
#     testing_score.append(np.mean(ind_test_score))


# In[45]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = training_data['store_and_fwd_flag']
values = np.array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)


# In[46]:


training_data.head()


# In[47]:


pd.get_dummies(data)


# In[48]:


final_train=training_data[['hour','weekday','trip_duration','Eucl_distance','store_and_fwd_flag']]


# In[49]:


final_train=pd.get_dummies(final_train)


# In[50]:


final_train


# In[51]:


final_train.info()


# In[52]:


testing_data.info()


# In[53]:


X=final_train[['hour','weekday','Eucl_distance','store_and_fwd_flag_N','store_and_fwd_flag_Y']]
y=final_train['trip_duration']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

X_train.shape, y_train.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


# In[54]:


X_test.shape, y_test.shape


# In[55]:


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.predict(X_test)


# In[56]:


def rmsle(y_test, y_pred) :
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


# In[57]:


rmsle(y_test,reg.predict(X_test))


# In[58]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=10)
neigh.fit(X_train, y_train)
neigh.predict(X_test)


# In[59]:


rmsle(y_test,neigh.predict(X_test))


# In[60]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=20)
neigh.fit(X_train, y_train)
neigh.predict(X_test)
rmsle(y_test,neigh.predict(X_test))


# In[61]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
rf.fit(X_train, y_train)
rf.predict(X_test)
rmsle(y_test,rf.predict(X_test))


# In[62]:


testing_data.pickup_datetime = testing_data.pickup_datetime.apply(convert_datetime)
testing_data['hour'] = testing_data.pickup_datetime.apply(get_hour)
testing_data['weekday'] = testing_data.pickup_datetime.apply(get_weekday)
testing_data['month'] = testing_data.pickup_datetime.dt.month
testing_data['Eucl_distance'] = testing_data.apply(distance, axis=1)


# In[63]:


testing_data.info()


# In[64]:


final_test= testing_data[['hour','weekday','Eucl_distance','store_and_fwd_flag']]


# In[65]:


final_test.head()


# In[66]:


final_test=pd.get_dummies(final_test)


# In[67]:


final_test.head()


# In[69]:


p=pd.Series(reg.predict(final_test))


# In[70]:


p


# In[71]:


submission = pd.DataFrame({
        "id":testing_data.id,
        "trip_duration": p
    })
submission.to_csv('submit_taxi.csv', index=False)

