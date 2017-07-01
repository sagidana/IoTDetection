import pandas as pd
from sklearn import preprocessing
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA


# Reading dataset
TRAINING_FILE_PATH = r"dataset\IoT_data_training.csv"
train_dataset = pd.read_csv(TRAINING_FILE_PATH)

TESTING_FILE_PATH = r"dataset\IoT_data_validation.csv"
test_dataset = pd.read_csv(TESTING_FILE_PATH)

# Preprocessing
def prepare_dataset(dataset):
	le = preprocessing.LabelEncoder()

	dataset.device_category = le.fit_transform(dataset.device_category.as_matrix())
	dataset.A_mac = le.fit_transform(dataset.A_mac.as_matrix())

	date_times = [datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S') for date_time in dataset.start.as_matrix()]
	date_times = [(d.hour * 60) + d.minute for d in date_times]

	dataset.start = date_times
	return dataset

def create_X_and_y(dataset, device_id):
	samples = dataset[dataset.device_category == device_id]
	
	anomaly_samples = dataset[dataset.device_category != device_id]
	anomaly_samples = anomaly_samples[:500]

	samples = pd.concat([samples, anomaly_samples], ignore_index=True)

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
Xs = []
ys = []
test_Xs = []
test_ys = []

for device_id in set(train_dataset.device_category):
	X, y = create_X_and_y(train_dataset, device_id)
	X_test, y_test = create_X_and_y(test_dataset, device_id)

	Xs.append(X)
	ys.append(y)
	test_Xs.append(X_test)
	test_ys.append(y_test)

# OneClassSVM
from sklearn import svm

for index in range(len(Xs)):
	# if (index != 0):
	# 	continue
	
	print(len(Xs[index]))

	# clf = svm.OneClassSVM(kernel="rbf", gamma=0.05, nu=0.0005) # For device category 2
	clf = svm.OneClassSVM(kernel="rbf", gamma=0.05, nu=0.0005)
	
	clf.fit(Xs[index])

	print("Results on training set:")
	y_pred = clf.predict(Xs[index])
	print("{}-> accuracy: {}\t percision: {}\t recall: {}".format(index, 
															accuracy_score(ys[index], y_pred), 
															precision_score(ys[index], y_pred),
															recall_score(ys[index], y_pred)))

	print("Results on testing set:")
	y_pred = clf.predict(test_Xs[index])
	print("{}-> accuracy: {}\t percision: {}\t recall: {}".format(index, 
															accuracy_score(test_ys[index], y_pred), 
															precision_score(test_ys[index], y_pred),
															recall_score(test_ys[index], y_pred)))

