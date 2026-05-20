import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
df=pd.read_csv("data.csv")
#print(df['diagnosis'].value_counts())
#print(df.duplicated().sum())
df.drop(['id','Unnamed: 32'],axis=1,inplace=True)
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
#corr=df.corr()['diagnosis'].sort_values(ascending=False)
#print(corr)
sns.countplot(df['diagnosis'])
plt.show()
X=df[['concave points_worst','perimeter_worst','radius_worst','area_worst']]
Y=df['diagnosis']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=SVC(kernel='linear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print(f"Confusion Matrix:{confusion_matrix(y_test,y_pred)}")
pickle.dump(model,open('model.pkl','wb'))
pickle.dump(scaler,open('scaler.pkl','wb'))


