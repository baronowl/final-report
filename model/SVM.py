
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import SVC


def data():
	X_train = np.load(
		"D:/#D/data/train_rri_ramp.npy")
	X_label = np.load("D:/#D/data/train_label.npy")
	Y_test = np.load("D:/#D/data/test_rri_ramp.npy")
	Y_label = np.load("D:/#D/data/test_label.npy")

	X_label = X_label.astype(dtype=np.int)
	Y_label = Y_label.astype(dtype=np.int)

	return X_train, X_label, Y_test, Y_label



x_train, y_train, x_test, y_test = data()
# a= x_train.shape
# print(a)


x_train_2D = (x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test_2D = (x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

# Find which kernel is the most fit one
# for k in [ 'rbf', 'sigmoid' ]:
#     clf = SVC(kernel=k)
#     clf.fit(x_train_2D, y_train)
#     confidence = clf.score(x_train_2D, y_train)
#     print("-------------------------")
#     print(k,confidence)


# file = open('test_record.txt', 'w')


best_C = None
best_gamma = None
best_score = 0
C=890000
# E = [100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]
F = [1,3,6,10,30,60,100,300,600,1000,3000,6000,10000,30000,60000,100000]
# for C in np.arange(1000000.0, 10000000.0, 1000000.0):



# clf = SVC(gamma='scale')
# clf.fit(X, y)
model = SVC(kernel='rbf', C=C)
model.fit(x_train_2D, y_train)
y_pred = model.predict(x_test_2D)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# gamma = 1 / (n_features * x_train_2D.var())
# Z = model.gamma
#
# print(Z)
# print(gamma)

#The following code is used to find the best paramter of gamma and C

# for i in range(len(F)):
# # for gamma in np.arange(1.0, 10.0, 1.0):
#     # file.write("\n--------------------------------------------------")
#     # C = E[i]
#     gamma = F[i]
#     model = SVC(kernel='rbf',  C=C,gamma=gamma)
#     model.fit(x_train_2D, y_train)
#     y_pred = model.predict(x_test_2D)
#     acc = metrics.accuracy_score(y_test, y_pred)
#         # score = model.score(x_test_2D, y_test)
#     # file.write("\nC= %.2f, gamma= %.2f, score= %.5f"%(C,gamma, acc))
#     # print("\nC= %.2f, gamma= %.2f, score= %.5f"%(C,gamma, acc))
#     # print("\nC= %.2f,  score= %.5f"%(C, acc))
#     print("\nC= %.2f, gamma= %.2f, score= %.5f"%(C,gamma, acc))
#         # if score > best_score:
#         #     best_score = score
#         #     best_C = C
#         #     best_gamma = gamma
# print('\n-----------------BEST------------------ ')
# print('\nHighest Accuracy Score: ', best_score)
# print('\nC= ', best_C)
# print('\ngamma=', best_gamma)
# file.close()

# Train the model using the training sets
# clf = SVC(kernel='sigmoid',C=1, gamma=0.5)
# clf.fit(x_train_2D, y_train)
# print("---------------")
#
#
# y_pred = clf.predict(x_test_2D)
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))