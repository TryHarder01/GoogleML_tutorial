# https://www.youtube.com/watch?v=84gqSbLcBFE&list=PLT6elRN3Aer7ncFlaCz8Zz-4B5cnsrOMt&index=5

from sklearn import datasets
iris = datasets.load_iris()


X  = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

from sklearn import tree

my_classifier  = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

#####

# try with KNeighbors

from sklearn.neighbors import KNeighborsClassifier
n_classifier = KNeighborsClassifier()
n_classifier.fit(X_train, y_train)

n_predictions = n_classifier.predict(X_test)

print(accuracy_score(y_test, n_predictions))
