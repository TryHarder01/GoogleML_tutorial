from sklearn import tree

print('hello-world')


features = [[ 140, 'smooth'], [130, 'smooth'], [150, 'bumpy'], [170, 'bumpy'] ]
labels = ['apple', 'apple', 'orange', 'orange']

# change features and labels to numerical data

features = [[ 140, 1], [130, 1], [150, 0], [170, 0] ]
labels = [0 , 0  , 1, 1]

# train a classifier... a decision tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features, labels)

print(clf.predict([[160, 0]]))
