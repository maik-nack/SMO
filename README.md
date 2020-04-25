# SMO
Implementation of a Support Vector Machine using the Sequential Minimal Optimization (SMO) algorithm for training.

# Algorithm

Implementation of Platt's SMO algorithm (Platt, John (1998). "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines") with improvements by Keerthi (S.S. Keerthi. S.K. Shevade. C. Bhattacharyya &. K.R.K. Murthy (2001). "Improvements to Piatt's SMO algorithm for SVM classifier design").

# Dependencies

##  For using SVM
- NumPy

# Documentation

## Setup model

Following parameters are default
```python
from SVM import SVM

model = SVM(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=1e-3, max_iter=-1)
```

Parameters have the same meaning as parameters in SVC of scikit-learn package ([link](https://scikit-learn.org/0.22/modules/generated/sklearn.svm.SVC.html)). But parameter `kernel` must be one of ‘linear’, ‘poly’, ‘rbf’ or a callable.

## Train model

This implementation only for two classes: `-1` and `1`.

Since Keerthi's article proposes two training methods, there are two modifications to this implementation.

First method:
```python
model.fit_modification1(X_train, y_train)
```
and second method:
```python
model.fit_modification2(X_train, y_train)
```

## Predict new observations

```python
y_hat = model.predict(X_test)
```

## Decision function

```python
decision = model.decision_function(X)
```
