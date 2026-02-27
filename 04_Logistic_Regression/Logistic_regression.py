# Q: Implement Logistic Regression to predict whether a candidate will be admitted based on:
# - GMAT Score
# - GPA
# - Work Experience
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df = pd.read_csv("candidates.csv")

X = df[['gmat', 'gpa', 'work_experience']]
y = df['admitted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

conf_matrix = pd.crosstab(
    y_test,
    y_pred,
    rownames=['Actual'],
    colnames=['Predicted']
)

print("Confusion Matrix:")
print(conf_matrix)

sn.heatmap(conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

new_candidates = pd.DataFrame({
    'gmat': [590, 740, 680, 610, 710, 650, 620, 720, 540, 700],
    'gpa': [2, 3.7, 3.3, 2.3, 3, 2, 3.1, 3.5, 2.6, 2.8],
    'work_experience': [3, 4, 6, 1, 5, 6, 5, 4, 2, 5]
})

new_pred = model.predict(new_candidates)

print("\nNew Candidates Data:")
print(new_candidates)

print("\nPredictions (0 = Not Admitted, 1 = Admitted):")
print(new_pred)
