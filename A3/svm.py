from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

#Load the breast cancer data set
cancer = load_breast_cancer()

#Split the data set into training/test sets
df = train_test_split(cancer.data, cancer.target, random_state=0)
X_train = df[0]
X_test = df[1]
y_train = df[2]
y_test = df[3]

#Run the default SVM w/ rbf(gaussian) kernel
svm = SVC()
svm.fit(X_train, y_train)
print svm
print('The accuracy on the training subset: {:.3f}'.format(svm.score(X_train, y_train)))
print('The accuracy on the test subset: {:.3f}'.format(svm.score(X_test, y_test)))
print ""

#Default model overfits, we need to scale data
min_train = X_train.min(axis=0)
range_train = (X_train - min_train).max(axis=0)

#Scale the x axis data points
X_train2 = (X_train - min_train)/range_train
X_test2 = (X_test - min_train)/range_train

#Run the SVM with polynomial kernel
svm = SVC(kernel='poly', degree=2)
svm.fit(X_train2, y_train)
print svm
print('The accuracy on the training subset: {:.3f}'.format(svm.score(X_train2, y_train)))
print('The accuracy on the test subset: {:.3f}'.format(svm.score(X_test2, y_test)))
print ""

#Run the SVM again with rbf(gaussian) kernel
svm = SVC(kernel='rbf', C=1000)
svm.fit(X_train2, y_train)
print svm
print('The accuracy on the training subset: {:.3f}'.format(svm.score(X_train2, y_train)))
print('The accuracy on the test subset: {:.3f}'.format(svm.score(X_test2, y_test)))
print ""