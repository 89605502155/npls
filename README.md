# npls

![PyPI](https://img.shields.io/pypi/v/npls?color=orange) ![Python 3.8, 3.9, 3.10, 3.11](https://img.shields.io/pypi/pyversions/npls?color=blueviolet) ![GitHub Pull Requests](https://img.shields.io/github/issues-pr/89605502155/N-PLS?color=blueviolet) ![License](https://img.shields.io/pypi/l/npls?color=blueviolet) ![Forks](https://img.shields.io/github/forks/89605502155/N-PLS?style=social)

**npls** - this module is a Python library for the N-PLS1 regression with L2-regularization.


## Installation

Install the current version with [PyPI](https://pypi.org/project/):

```bash
pip install npls
```

Or from Github:
```bash
pip install https://github.com/89605502155/N-PLS/main.zip
```

## Usage

You can fit your own regression model. n_components - is a number of components of SVD decomposition and a is a parameter of L2-regularization. X_train - is a 3d-array. y_train -is a vector.

```python
from npls import npls 

model=npls(n_components=4, a=0.09)
model.fit(X_train,y_train)
#components of svd-decomposition
w_i=model.w_i
w_k=model.w_k
#predict
y_predicted=model.predict(X_test)
```

## Example

You can use this library with Scikit-learn library. For example, we can use GridSearchCV.

*If you installed a module from PyPi, you should to import it like this: ``` from npls import npls  ```*

*If from GitHub or source: ``` from npls import npls ```*

```python
from npls import npls 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, make_scorer
import sklearn
from sklearn.model_selection import GridSearchCV

npls1=npls()
#you can use many error metrics
scoring={'mse': make_scorer(mean_squared_error),'r2':'r2'}
parametrsNames={'n_components': [4],
                'a': np.logspace(-25, 25,num = 51)}

gridCought=GridSearchCV(npls1, parametrsNames, cv=5, 
    scoring=scoring,refit='r2',return_train_score=True)
gridCought.fit(X_train,y_train)
#errors
r2_p=gridCought.score(X_test.copy(), y_test.copy())
mse_cv=gridCought.cv_results_[ "mean_test_mse" ]
mse_c=gridCought.cv_results_[ "mean_train_mse" ]
r2_cv=gridCought.cv_results_[ "mean_test_r2" ]
r2_c=gridCought.cv_results_[ "mean_train_r2" ]
```
