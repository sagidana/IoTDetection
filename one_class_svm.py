import pandas as pd
from sklearn import preprocessing
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt


# Reading dataset
TRAINING_FILE_PATH = r"dataset\IoT_data_training.csv"
train_dataset = pd.read_csv(TRAINING_FILE_PATH)

TESTING_FILE_PATH = r"dataset\IoT_data_validation.csv"
test_dataset = pd.read_csv(TESTING_FILE_PATH)

labeler = None

# Preprocessing
def prepare_dataset(dataset):
	global labeler
	labeler = preprocessing.LabelEncoder()
	le = preprocessing.LabelEncoder()

	dataset.device_category = labeler.fit_transform(dataset.device_category.as_matrix())
	dataset.A_mac = le.fit_transform(dataset.A_mac.as_matrix())

	date_times = [datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S') for date_time in dataset.start.as_matrix()]
	date_times = [(d.hour * 60) + d.minute for d in date_times]

	dataset.start = date_times
	return dataset

def create_X_and_y(dataset, device_id, anomaly_ratio=0.05):
	samples = dataset[dataset.device_category == device_id]
	samples_1 = []
	anomaly_samples = dataset[dataset.device_category != device_id]
	num_of_anomalies = int(anomaly_ratio*len(samples))
	for x in range(num_of_anomalies):
		rand_device = random.randrange(0,len(set(dataset.device_category)),1)

		while rand_device == device_id:
			rand_device = random.randrange(0,len(set(dataset.device_category)),1)

		other_device_samples = dataset[dataset.device_category == rand_device]
		
		random_sample_index = random.randrange(0,len(other_device_samples),1)

		samples.loc[len(samples)] = other_device_samples.iloc[random_sample_index,:]
		# samples.append(other_device_samples.iloc[random_sample_index,:], ignore_index=True)

	X = samples.drop(['mac_start_index', 'overall_index', 'device_category', 'A_mac'], axis=1)
	
	# Scaling
	mms = preprocessing.MinMaxScaler()
	X = mms.fit_transform(X)

	# Dimentional reduction
	pca = PCA(n_components=5)
	X = pca.fit_transform(X)

	y = samples.device_category.as_matrix()
	y = [1 if label == device_id else -1 for label in y]

	return X, y

train_dataset = prepare_dataset(train_dataset)
test_dataset = prepare_dataset(test_dataset)

# Create datasets
Xs = {}
ys = {}
test_Xs = {}
test_ys = {}

for device_id in set(train_dataset.device_category):
	X, y = create_X_and_y(train_dataset, device_id)
	X_test, y_test = create_X_and_y(test_dataset, device_id)

	Xs[labeler.inverse_transform([device_id])[0]] = X
	ys[labeler.inverse_transform([device_id])[0]] = y
	test_Xs[labeler.inverse_transform([device_id])[0]] = X_test
	test_ys[labeler.inverse_transform([device_id])[0]] = y_test

# OneClassSVM
from sklearn import svm

def create_classifier(X, y):
	chosen_gamma = 0.5
	chosen_nu = 0.005
	max_precision = 0
	for gamma in [0.5,0.2,0.1,0.05,0.005]:
		for nu in [0.1,0.01,0.005,0.0005,0.00005]:
			clf = svm.OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
			clf.fit(X)
			y_pred = clf.predict(X)
			current_precision = accuracy_score(y, y_pred)
			# current_precision = precision_score(y, y_pred)
			if current_precision > max_precision:
				chosen_gamma = gamma
				chosen_nu = nu
				max_precision = current_precision

	clf = svm.OneClassSVM(kernel="rbf", gamma=chosen_gamma, nu=chosen_nu)
	clf.fit(X)
	return clf


def create_classifiers(Xs, ys, test_Xs, test_ys):
	clfs = []
	for index in Xs:
		# clf = svm.OneClassSVM(kernel="rbf", gamma=0.05, nu=0.0005) # For device category 2
		clf = create_classifier(Xs[index], ys[index])

		clfs.append(clf)

		print("{} Classifier".format(index))
		
		print("Results on training set:")
		print("Number of samples: {}".format(len(Xs[index])))
		y_pred = clf.predict(Xs[index])
		print("{}-> accuracy: {}\t percision: {}\t recall: {}".format(index, 
																accuracy_score(ys[index], y_pred), 
																precision_score(ys[index], y_pred),
																recall_score(ys[index], y_pred)))

		print("Results on testing set:")
		print("Number of samples: {}".format(len(test_Xs[index])))
		y_pred = clf.predict(test_Xs[index])
		print("{}-> accuracy: {}\t percision: {}\t recall: {}".format(index, 
																accuracy_score(test_ys[index], y_pred), 
																precision_score(test_ys[index], y_pred),
																recall_score(test_ys[index], y_pred)))

		# Plot roc
		scoring = clf.decision_function(test_Xs[index])

		fpr, tpr, _ = roc_curve(test_ys[index], scoring)
		roc_auc = auc(fpr, tpr)

		plt.figure()
		lw = 2
		plt.plot(fpr, tpr, color='darkorange',
	 				lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic example')
		plt.legend(loc="lower right")
		plt.show()

	return clfs

clfs = create_classifiers(Xs,ys,test_Xs,test_ys)

# preds = []
# for index in range(len(clfs)):
# 	preds.append(clfs[index].decision_function([test_Xs[4][10]]))

# print(preds)





