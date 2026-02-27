# Q: Implement a Decision Tree classifier using the ID3 algorithm to predict whether a person has diabetes.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree

df = pd.read_csv("diabetes.csv")

X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=1,
    stratify=y
)

id3_clf = DecisionTreeClassifier(
    criterion="entropy",   
    max_depth=3,
    min_samples_leaf=5,
    random_state=100
)

id3_clf.fit(X_train, y_train)

y_pred = id3_clf.predict(X_test)

print("Predicted Output:")
print(y_pred)

print("\nID3 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

plt.figure(figsize=(20, 10))
tree.plot_tree(
    id3_clf,
    feature_names=X.columns,
    class_names=["No Diabetes", "Diabetes"],
    filled=True
)
plt.title("ID3 Decision Tree (Entropy Based)")
plt.show()
