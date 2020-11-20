"""Goal:
Import dataset
Train a classifier
Predict label for new flower
Visualise the tree
"""

import pandas as pd
from sklearn import tree
import graphviz
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


iris = pd.read_csv("./datasets/iris.csv")

# setting the dependent und independent variables
x = iris.iloc[:, 0:4].astype(float)
y = iris["variety"]


print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=10)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# prediction for a new flowers
y_pred = clf.predict(x_test)

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=x_test.columns.values,
                                class_names=y_test.values,
                                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("./export/iris")
