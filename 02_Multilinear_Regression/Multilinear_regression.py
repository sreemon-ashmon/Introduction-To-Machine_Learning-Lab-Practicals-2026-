## Q:Implement Multiple Linear Regression to predict Stock Index Price:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

file_path = "stock_multiple_regression.csv"
df = pd.read_csv(file_path)

X = df[['Interest_Rate', 'Unemployment_Rate']]
Y = df['Stock_Index_Price']

regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Intercept:', regr.intercept_)
print('Coefficients:', regr.coef_)

Y_pred = regr.predict(X)

plt.scatter(Y, Y_pred)
plt.xlabel('Actual Stock Index Price')
plt.ylabel('Predicted Stock Index Price')
plt.title('Actual vs Predicted Stock Index Price')
plt.grid(True)
plt.show()
