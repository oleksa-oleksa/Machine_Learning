import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

boston = pd.read_csv("./datasets/BostonHousing.csv")

# setting the dependent und independent variables
x = boston.iloc[:,0:13]
y = boston["medv"]

# Compute pairwise correlation of columns, excluding NA/null values.
names = []
correlations = boston.corr()

# Plot rectangular data as a color-encoded matrix.
sns.heatmap(correlations, square=True, cmap="YlGnBu")
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=10)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Create linear regression object
# Train the model using the training sets
regr = LinearRegression().fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# compare prediction with test data
fig, ax = plt.subplots()
ax2 = ax.twinx()
sns.lineplot(data=y_test, color='r')
sns.lineplot(data=y_pred, color='b', ax=ax2)
plt.show()
