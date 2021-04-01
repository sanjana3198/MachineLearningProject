#importing library
import numpy as np
import matplotlib.pyplot as plt

#fetching MNIST Dataset
from sklearn.datasets import fetch_openml
dataset = fetch_openml('mnist_784', version = 1)

#dataset.data[0]

X = dataset.data
y = dataset.target
y = y.astype(np.number)

#To show the image of a particular handwritten digit
plt.imshow(X[4098].reshape(28, 28), cmap = 'binary') #Here I have selected 9
plt.show()

#To show the image of a different handwritten digits
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X[i].reshape(28, 28), cmap = 'binary')
    plt.axis('off')
    plt.xlabel(y[i])
plt.show()

#Train Test Split so as to proceed with model building
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Classifiction

#Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

#To check the accuracy of log_reg on train data
log_reg.score(X_train, y_train)

#To check the accuracy of log_reg on test data
log_reg.score(X_test, y_test)

#Applying Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()
dtf.fit(X_train, y_train)

#To check the accuracy of dtf on train data
dtf.score(X_train, y_train)

#To check the accuracy of dtf on test data
dtf.score(X_test, y_test)

#Applying Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

#To check the accuracy of nb on train data
nb.score(X_train, y_train)

#To check the accuracy of nb on test data
nb.score(X_test, y_test)

#Probability of target using log_reg
y_pred = log_reg.predict(X_test)

#predictions
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i].reshape(28, 28), cmap = 'binary')
    plt.xlabel("Actual label : {} \nPredicted label : {}".format(y_test[i], y_pred[i]))
plt.tight_layout()
plt.show()

#Probability of target using dtf
y_pred_dtf = dtf.predict(X_test)

#predictions
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i].reshape(28, 28), cmap = 'binary')
    plt.xlabel("Actual label : {} \nPredicted label : {}".format(y_test[i], y_pred_dtf[i]))
plt.tight_layout()
plt.show()

#To check the confusin matrix for log_reg
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

cm = confusion_matrix(y_test, y_pred)

precision_score(y_test, y_pred, average = 'micro')
recall_score(y_test, y_pred, average = 'micro')
f1_score(y_test, y_pred, average = 'micro')

#Cross validation for log_reg
from sklearn.model_selection import cross_val_score
cross_val_score(log_reg, X_train, y_train, cv = 3)

#KFold for log_reg
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)

scores = []

for train_idx, test_idx in kf.split(X_train):
    print(train_idx, test_idx)
    X_train_idx, X_test_idx, y_train_idx, y_test_idx = X_train[train_idx], X_train[test_idx], y_train[train_idx], y_train[test_idx]
    log_reg.fit(X_train_idx, y_train_idx)
    scores.append(log_reg.score(X_test_idx, y_test_idx))

scores

#LeaveOneOut for log_reg
from sklearn.model_selection import LeaveOneOut
loocv = LeaveOneOut()

scores_loocv = []

for train_idx, test_idx in loocv.split(X_train):
    print(train_idx, test_idx)
    X_train_idx, X_test_idx, y_train_idx, y_test_idx = X_train[train_idx], X_train[test_idx], y_train[train_idx], y_train[test_idx]
    log_reg.fit(X_train_idx, y_train_idx)
    scores_loocv.append(log_reg.score(X_test_idx, y_test_idx))

scores_loocv

#Hyper-Parameters for GridSearchCV
params = {'criterion': ['gini', 'entropy'],
          'max_depth': [5, 6, 7]}

#GridSearchCV
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(log_reg, params)
grid.fit(X_train, y_train)

#best parameters
grid.best_params_

#best estimator
best_log_reg = grid.best_estimator_

best_log_reg.score(X_test, y_test)

#Probability of target using rf
y_pred_best_log_reg = best_log_reg.predict(X_test)

#predictions
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i].reshape(28, 28), cmap = 'binary')
    plt.xlabel("Actual label : {} \nPredicted label : {}".format(y_test[i], y_pred_best_log_reg[i]))
plt.tight_layout()
plt.show()

#Applying Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

#To check the accuracy of rf on train data
rf.score(X_train, y_train)

#To check the accuracy of rf on test data
rf.score(X_test, y_test)

#Probability of target using rf
y_pred_rf = rf.predict(X_test)

#predictions
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i].reshape(28, 28), cmap = 'binary')
    plt.xlabel("Actual label : {} \nPredicted label : {}".format(y_test[i], y_pred_rf[i]))
plt.tight_layout()
plt.show()

#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
X_decom = pca.fit_transform(X)

#Train test Split
from sklearn.model_selection import train_test_split
X_train_decom, X_test_decom, y_train, y_test = train_test_split(X_decom, y)

#Applying Random Forest Classifier
rf_decom = RandomForestClassifier()
rf_decom.fit(X_train_decom, y_train)

#To check the accuracy of rf_decom on train data
rf.score(X_train_decom, y_train)

#To check the accuracy of rf_decom on test data
rf.score(X_test_decom, y_test)

#Probability of target using dtf
y_pred_rf_decom = rf_decom.predict(X_test)

#predictions
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i].reshape(28, 28), cmap = 'binary')
    plt.xlabel("Actual label : {} \nPredicted label : {}".format(y_test[i], y_pred_rf_decom[i]))
plt.tight_layout()
plt.show()