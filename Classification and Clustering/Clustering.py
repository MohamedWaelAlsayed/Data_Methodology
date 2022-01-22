import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
aftervisualization = pd.read_csv("DS project.csv",)

dataset=aftervisualization.copy()
dataset.head()
encoder = preprocessing.LabelEncoder() 
#transform coulms to numbers 
dataset['hotel'] = encoder.fit_transform(dataset['hotel'])
dataset['meal'] = encoder.fit_transform(dataset['meal'])
dataset['country'] = encoder.fit_transform(dataset['country'])
dataset['market_segment'] = encoder.fit_transform(dataset['market_segment'])
dataset['distribution_channel'] = encoder.fit_transform(dataset['distribution_channel'])
dataset['reserved_room_type'] = encoder.fit_transform(dataset['reserved_room_type'])
dataset['assigned_room_type'] = encoder.fit_transform(dataset['assigned_room_type'])
dataset['deposit_type'] = encoder.fit_transform(dataset['deposit_type'])
dataset['customer_type'] = encoder.fit_transform(dataset['customer_type'])
dataset['reservation_status'] = encoder.fit_transform(dataset['reservation_status'])
dataset['reservation_status_date'] = encoder.fit_transform(dataset['reservation_status_date'])
dataset['arrival_date'] = encoder.fit_transform(dataset['arrival_date'])

model_of_kmeans = KMeans(n_clusters=15)

#selecting attributes
X = dataset.iloc[:,:]

Y = dataset.iloc[:,0:1]

#splitiing data
X_train, X_test, y_train, y_test = train_test_split(
     X, Y, test_size=0.05, random_state=42)

model_of_kmeans.fit(X_train)

print("score of training model : ",model_of_kmeans.score(X_train))
print("score of testing model : ",model_of_kmeans.score(X_test))
centers = model_of_kmeans.cluster_centers_
print("centers of model : ",centers)
print("labels are : ",model_of_kmeans.labels_)
print('number of iteration : ', model_of_kmeans.n_iter_)

#conclude prediction
y_predicted = model_of_kmeans.predict(X_test)
print('some predicted values of kmeansmodel : ',y_predicted[:15])

#visualization
plt.scatter(X_train.iloc[:,0],X_train.iloc[:,1],c='black',s=200)
plt.scatter(X_test.iloc[:,0],X_test.iloc[:,1],c='b',s=30)
for i in range(len(centers)):
    plt.scatter(centers[i,0],centers[i,1],c='r',s=200)
plt.show()