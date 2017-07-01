import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
# from matplotlib.pylab import rcParams

# rcParams['figure.figsize'] = 12, 4


TEST_FILE_PATH = r"IoT_data_test.csv"
TRAINING_FILE_PATH = r"IoT_data_training.csv"
VALIDATION_FILE_PATH = r"IoT_data_validation.csv"
target = 'device_category'

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "Model Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


csv_data_train = pd.read_csv(TRAINING_FILE_PATH)
csv_data_train.device_category
le = preprocessing.LabelEncoder()

train_labels = csv_data_train.as_matrix()[:,-3]
train_labels = le.fit_transform(train_labels)

csv_data_train.device_category = train_labels

# train_samples = csv_data_train.as_matrix()[:,:-5]

# dtrain = xgb.DMatrix(train_samples, label=train_labels)

# csv_data_validation = pd.read_csv(VALIDATION_FILE_PATH)

# test_labels = csv_data_validation.as_matrix()[:,-3]
# test_labels = le.fit_transform(test_labels)

# validation_samples = csv_data_validation.as_matrix()[:,:-5]

# dtest = xgb.DMatrix(validation_samples, label=test_labels)

# param = {'max_depth':4, 'eta':0.1 }
# param['eval_metric'] = 'error'

# evallist  = [(dtest,'eval'), (dtrain,'train')]

# bst = xgb.train(param, dtrain, 10, evallist)
temp = csv_data_train.copy()

# for chosen_one in xrange(3):
chosen_one = 1
compared_to = 5
print(chosen_one)

csv_data_train = temp.copy()

csv_data_train = csv_data_train[(csv_data_train.device_category == chosen_one) | (csv_data_train.device_category == compared_to)]

labels = csv_data_train.device_category.as_matrix()

for index in range(len(labels)):
	if labels[index] == chosen_one:
		labels[index] = 123

for index in range(len(labels)):
	if labels[index] != 123:
		labels[index] = 0

for index in range(len(labels)):
	if labels[index] == 123:
		labels[index] = 1

csv_data_train.device_category = labels

print(len(csv_data_train))

predictors = [x for x in csv_data_train.columns if x not in [target, 'start', 'overall_index', 'mac_start_index','A_mac']]

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=8,
 scale_pos_weight=1,
 seed=27)
#xgb1.n_classes_=11
modelfit(xgb1, csv_data_train, predictors)
