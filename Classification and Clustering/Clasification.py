import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
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

#selecting attributes
X = dataset.iloc[:,1:]
#selecting hotel attribute as output
Y = dataset.iloc[:,0:1]

#splitiing data
X_train, X_test, y_train, y_test = train_test_split(
     X, Y, test_size=0.0009, random_state=42)
 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print('score of training : ',clf.score(X_train,y_train))
print('score of testing : ',clf.score(X_test,y_test))
print("----------------------------")
print('percentage of importance of features : ',clf.feature_importances_)

#conclude prediction

y_predicted = clf.predict(X_test)

#visualization
#making cm to show correlation between what we predicted an our original data
cofficent_matrix = confusion_matrix(y_test,y_predicted)

sns.heatmap(cofficent_matrix,center=True,annot=True, fmt="d")

plt.show()











 