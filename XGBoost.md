For sure we have to import some libraries :D
```python
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```

I first import the necessary libraries: `xgboost`, which is the main library for building XGBoost models; `load_iris` from `sklearn.datasets`, which allows us to load the Iris dataset; `accuracy_score` from `sklearn.metrics`, which will be used to calculate the accuracy of our predictions; and `train_test_split` from `sklearn.model_selection`, which we'll use to split the data into training and testing sets.

```python
iris = load_iris()
```
Load the Iris dataset into a variable called `iris`.

```python
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=123)
```

Then split the data into training and testing sets using `train_test_split`. I pass in the data (`iris.data`) and labels (`iris.target`) as arguments, along with the test size (`test_size=0.2`) and a random state (`random_state=123`) to ensure reproducibility.

```python
xg_clf = xgb.XGBClassifier(objective ='multi:softmax', num_class=3, colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
```

I instantiate an XGBoost classifier model using `xgb.XGBClassifier`. Here, we're setting various hyperparameters for the model, including the objective function (`'multi:softmax'`, since this is a multiclass classification problem), the number of classes (`num_class=3`), the fraction of columns to be randomly sampled for each tree (`colsample_bytree = 0.3`), the learning rate (`learning_rate = 0.1`), the maximum depth of each tree (`max_depth = 5`), the L1 regularization term (`alpha = 10`), and the number of trees (`n_estimators = 10`).

```python
xg_clf.fit(X_train,y_train)
```

Train the XGBoost classifier model on the training data using `fit`.

```python
preds = xg_clf.predict(X_test)
```
Then I use the trained model to make predictions on the test data using `predict`.

```python
accuracy = accuracy_score(y_test, preds)
print("Accuracy: %f" % (accuracy))
```

I calculate the accuracy of the predictions using `accuracy_score`, passing in the true labels (`y_test`) and predicted labels (`preds`), and then print out the accuracy as a floating point value using `print`.


Finally the output would be like:

`Accuracy: 0.986667`

So also we can do some plottings:

```python
import matplotlib.pyplot as plt

# plot feature importance
xgb.plot_importance(xg_clf)
plt.show()
```

![Untitled](https://user-images.githubusercontent.com/109058050/230643400-8953c53e-9032-401f-acaa-be906c5d9a09.png)



```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# calculate confusion matrix
cm = confusion_matrix(y_test, preds)

# plot confusion matrix heatmap
sns.heatmap(cm, annot=True, cmap="Blues")
plt.show()
```

![Untitled](https://user-images.githubusercontent.com/109058050/230643438-88d382c1-686c-453a-b8fa-58641b97a208.png)

Also for the iterations:

```python
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load iris data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# set parameters
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 3,
    'eta': 0.1,
    'gamma': 0.1,
    'lambda': 1,
    'alpha': 1,
    'eval_metric': ['merror', 'mlogloss']
}

# fit model
evolution = {}
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100, evals=[(dtrain, 'train'), (dtest, 'test')], evals_result=evolution)

# plot training and validation error metrics
plt.plot(evolution['train']['mlogloss'], label='train_logloss')
plt.plot(evolution['test']['mlogloss'], label='test_logloss')
plt.plot(evolution['train']['merror'], label='train_error')
plt.plot(evolution['test']['merror'], label='test_error')
plt.legend()
plt.show()
```

![Untitled](https://user-images.githubusercontent.com/109058050/230643468-e600f35c-7e0f-4407-9933-396d2085a192.png)


