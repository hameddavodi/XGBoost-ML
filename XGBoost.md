Sure, here's a detailed explanation of each line of the code:

```python
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```

We first import the necessary libraries: `xgboost`, which is the main library for building XGBoost models; `load_iris` from `sklearn.datasets`, which allows us to load the Iris dataset; `accuracy_score` from `sklearn.metrics`, which will be used to calculate the accuracy of our predictions; and `train_test_split` from `sklearn.model_selection`, which we'll use to split the data into training and testing sets.

```python
iris = load_iris()
```

We load the Iris dataset into a variable called `iris`.

```python
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=123)
```

We split the data into training and testing sets using `train_test_split`. We pass in the data (`iris.data`) and labels (`iris.target`) as arguments, along with the test size (`test_size=0.2`) and a random state (`random_state=123`) to ensure reproducibility.

```python
xg_clf = xgb.XGBClassifier(objective ='multi:softmax', num_class=3, colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
```

We instantiate an XGBoost classifier model using `xgb.XGBClassifier`. Here, we're setting various hyperparameters for the model, including the objective function (`'multi:softmax'`, since this is a multiclass classification problem), the number of classes (`num_class=3`), the fraction of columns to be randomly sampled for each tree (`colsample_bytree = 0.3`), the learning rate (`learning_rate = 0.1`), the maximum depth of each tree (`max_depth = 5`), the L1 regularization term (`alpha = 10`), and the number of trees (`n_estimators = 10`).

```python
xg_clf.fit(X_train,y_train)
```

We train the XGBoost classifier model on the training data using `fit`.

```python
preds = xg_clf.predict(X_test)
```

We use the trained model to make predictions on the test data using `predict`.

```python
accuracy = accuracy_score(y_test, preds)
print("Accuracy: %f" % (accuracy))
```

We calculate the accuracy of the predictions using `accuracy_score`, passing in the true labels (`y_test`) and predicted labels (`preds`), and then print out the accuracy as a floating point value using `print`.
