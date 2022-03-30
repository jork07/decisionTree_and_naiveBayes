import pandas as pd
from sklearn import tree, metrics, model_selection

# load data set
header = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv('car.data', names=header)

# target variable
data['class'], class_names = pd.factorize(data['class'])

# identify predictor variables to integer codes
data['buying'],_ = pd.factorize(data['buying'])
data['maint'],_ = pd.factorize(data['maint'])
data['doors'],_ = pd.factorize(data['doors'])
data['persons'],_ = pd.factorize(data['persons'])
data['lug_boot'],_ = pd.factorize(data['lug_boot'])
data['safety'],_ = pd.factorize(data['safety'])

# select predictor and target
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# train test splits
x_train_90, x_test_10, y_train_90, y_test_10 = model_selection.train_test_split(x, y, test_size=0.1, random_state=0)
x_train_80, x_test_20, y_train_80, y_test_20 = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)
x_train_70, x_test_30, y_train_70, y_test_30 = model_selection.train_test_split(x, y, test_size=0.3, random_state=0)
x_train_60, x_test_40, y_train_60, y_test_40 = model_selection.train_test_split(x, y, test_size=0.4, random_state=0)
x_train_50, x_test_50, y_train_50, y_test_50 = model_selection.train_test_split(x, y, test_size=0.5, random_state=0)

# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

# results 90/10 split
dtree.fit(x_train_90, y_train_90)
y_pred_10 = dtree.predict(x_test_10)
count_misclassified = (y_test_10 != y_pred_10).sum()
print('90/10 Split')
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test_10, y_pred_10)
print('Accuracy: {:.3f}'.format(accuracy))

dtree.fit(x_train_80, y_train_80)
y_pred_20 = dtree.predict(x_test_20)
count_misclassified = (y_test_20 != y_pred_20).sum()
print('80/20 Split')
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test_20, y_pred_20)
print('Accuracy: {:.3f}'.format(accuracy))

dtree.fit(x_train_70, y_train_70)
y_pred_30 = dtree.predict(x_test_30)
count_misclassified = (y_test_30 != y_pred_30).sum()
print('70/30 Split')
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test_30, y_pred_30)
print('Accuracy: {:.3f}'.format(accuracy))

# results 60/40 split
dtree.fit(x_train_60, y_train_60)
y_pred_40 = dtree.predict(x_test_40)
count_misclassified = (y_test_40 != y_pred_40).sum()
print('60/40 Split')
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test_40, y_pred_40)
print('Accuracy: {:.3f}'.format(accuracy))

# results 50/50 split
dtree.fit(x_train_50, y_train_50)
y_pred_50 = dtree.predict(x_test_50)
count_misclassified = (y_test_50 != y_pred_50).sum()
print('50/50 Split')
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test_50, y_pred_50)
print('Accuracy: {:.3f}'.format(accuracy))