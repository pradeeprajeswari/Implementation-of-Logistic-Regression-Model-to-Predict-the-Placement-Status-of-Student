# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1:Start

STEP 2:Load and preprocess the dataset: drop irrelevant columns, handle missing values, and encode categorical variables using LabelEncoder.

STEP 3:Split the data into training and test sets using train_test_split.

STEP 4:Create and fit a logistic regression model to the training data.

STEP 5:Predict the target variable on the test set and evaluate performance using accuracy, confusion matrix, and classification report.

STEP 6:Display the confusion matrix using metrics.ConfusionMatrixDisplay and plot the results.

STEP 7:End

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Pradeep E
RegisterNumber:  212223230149
*/

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
PLACEMENT DATA
![Screenshot 2024-09-20 102317](https://github.com/user-attachments/assets/8f6ba3f0-6f90-4620-971a-7c3bc07f7442)

SALARY DATA
![Screenshot 2024-09-20 102326](https://github.com/user-attachments/assets/60fd4ee6-cfa8-4a95-80b3-dd7d7c6171bd)

![Screenshot 2024-09-20 102335](https://github.com/user-attachments/assets/271f38ff-07d3-4548-b0f9-249937e66cf5)

![Screenshot 2024-09-20 102346](https://github.com/user-attachments/assets/ec1f8ff2-7f32-4bec-9dc7-af1959c5bb6d)

![Screenshot 2024-09-20 102404](https://github.com/user-attachments/assets/430108aa-1ca4-49f5-8ac5-3e4a82768167)

![Screenshot 2024-09-20 102412](https://github.com/user-attachments/assets/dcab0ab8-df9f-440b-a0a0-c875e2e3fbac)

![Screenshot 2024-09-20 102420](https://github.com/user-attachments/assets/48e6e5fd-6693-4ac3-960b-31faf495b85b)

![Screenshot 2024-09-20 102426](https://github.com/user-attachments/assets/43454d0b-6897-4ef4-bfbd-183b41da8132)

![Screenshot 2024-09-20 102445](https://github.com/user-attachments/assets/2dffe670-12da-4012-b310-6b0ff5103a11)

![Screenshot 2024-09-20 102450](https://github.com/user-attachments/assets/622dc472-f4fb-44d2-a695-43a7675b5b70)

![Screenshot 2024-09-20 102458](https://github.com/user-attachments/assets/854ecd21-7ab0-41df-91e5-ce1e382364da)

PREDICTION OF LR

![Screenshot 2024-09-20 102508](https://github.com/user-attachments/assets/edd3060d-7fb8-4acc-9a30-5d769cf27dca)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
