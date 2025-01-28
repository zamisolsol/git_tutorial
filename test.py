import numpy as np
from moabb.datasets import Cho2017
from moabb.paradigms import LeftRightImagery
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load MOABB dataset
dataset = Cho2017()
paradigm = LeftRightImagery(resample=128)

# Get data from the first subject and the first session

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[6])

C3_index = 12
C4_index = 53

C3_values = X[:, C3_index, :]
C4_values = X[:, C4_index, :]

C3_value_mean = C3_values.mean(axis=1)
C4_values_mean = C4_values.mean(axis=1)
C3_value_std = C3_values.std(axis=1)
C4_value_std = C4_values.std(axis=1)
C3_sub_C4 = C3_values - C4_values


all_input = np.column_stack((
    C3_value_mean, C4_values_mean, C3_value_std, C4_value_std, C3_sub_C4))

X_train, X_test, y_train, y_test = train_test_split(all_input, y , test_size = 0.3, random_state = 42)
clf = LogisticRegression()
clf.fit(X_train, y_train)

predict = clf.predict(X_test)
accuracy = accuracy_score(y_test, predict)
print(f"정확도 (Accuracy): {accuracy:.2f}")