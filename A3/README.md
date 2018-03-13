# CP640 - Assignment 3 
### Ryan Kazmerik (175826410)

## Question 3
Please use Weka/LIBSVM or your favorite programming language to implement
a support vector machine with both Polynomial and Gaussian kernels. Exploit
the UCI Breast Cancer Dataset 1 to compare their performance. Hand in your
code and complete analysis based on the empirical study.

## Results
1. SVM - Polynomial Kernel
<pre>
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=2, gamma='auto', kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

The accuracy on the training subset: 0.810
The accuracy on the test subset: 0.818
</pre>

2. SVM - Gaussian Kernel (RBF)
<pre>
SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

The accuracy on the training subset: 0.988
The accuracy on the test subset: 0.972
</pre>

## Explanation
The polynomial kernel uses a polynomial function to project the dataset into a feature space with a high dimensionality.

However, the gaussian kernel uses an exponetial function to project the dataset into a high dimensionality feature space. This method produces a feature space that is linearly seperable - and if we consider our dataset, it is also linearly seperable as it is a binary classification (ex. class = malignant or benign)

Therefore the gaussian kernel demonstrates increased acccuracy, as the feature projection is more appropriate for our dataset.


