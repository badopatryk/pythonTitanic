import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
# print(data.isnull().sum())  # Age: 177, Cabin: 687, Embarked: 2

Y = data['Survived']

data = data.drop(columns=['PassengerId', 'Name',
                 'Ticket', 'Cabin', 'Survived'])
test_data = test_data.drop(
    columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {
             'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

test_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {
                  'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

data['Age'].fillna(data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

# changing missing values to S
data['Embarked'].fillna(0, inplace=True)
test_data['Embarked'].fillna(0, inplace=True)

X_train, X_test, Y_train, Y_test = train_test_split(
    data, Y, test_size=0.3, random_state=1)

# print(data.head())

#model = DecisionTreeRegressor(random_state=1)
model = LogisticRegression(random_state=1)

model.fit(X_train, Y_train)
#val_pred = model.predict(X_test)
print("Training data: ", model.score(X_train, Y_train))
print("Testing data: ", model.score(X_test, Y_test))

test_data_2 = test_data.iloc[random.randint(0, len(test_data))]
test_data_2 = np.asarray(test_data_2).reshape(1, -1)

pred = model.predict(test_data_2)

# print(test_data.columns)
# print(data.columns)

if pred[0]:
    print("Alive")
else:
    print("Dead")
