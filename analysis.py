__author__ = 'Chao'

import numpy as np
from sklearn import svm, cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

activity_label = {'1': 'WALKING',
                  '2': 'WALKING_UPSTAIRS',
                  '3': 'WALKING_DOWNSTAIRS',
                  '4': 'SITTING',
                  '5': 'STANDING',
                  '6': 'LAYING'}

# ############################# Open data set ###############################
X = []
y = []
X_fin = []
y_fin = []

print "Opening dataset..."
try:
    with open("X_train.txt", 'rU') as f:
        res = list(f)
        for line in res:
            line.strip("\n")
            pair = line.split(" ")
            while pair.__contains__(""):
                pair.remove("")
            for i in xrange(pair.__len__()):
                pair[i] = float(pair[i])
            X.append(pair)
        f.close()
    with open("y_train.txt", 'rU') as f:
        res = list(f)
        for line in res:
            y.append(int(line.strip("\n")[0]))
        f.close()
except:
    print "Error in reading the train set file."
    exit()
try:
    with open("X_test.txt", 'rU') as f:
        res = list(f)
        for line in res:
            line.strip("\n")
            pair = line.split(" ")
            while pair.__contains__(""):
                pair.remove("")
            for i in xrange(pair.__len__()):
                pair[i] = float(pair[i])
            X_fin.append(pair)
        f.close()
    with open("y_test.txt", 'rU') as f:
        res = list(f)
        for line in res:
            y_fin.append(int(line.strip("\n")[0]))
        f.close()
except:
    print "Error in reading the train set file."
    exit()
print "Dataset opened."

X = np.array(X)
y = np.array(y)


###### Separate data set into 70% training set and 30% test set
print "Separating data into 70% training set & 30% test set..."
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
print "Dataset separated."

###### Get best parameters ######
############################### Kernel=Linear ###############################
print "######## SVM, Kernel = Linear #########"

#C_linear = [0.1, 1, 10, 100]
C_linear = [3]

result_linear = []

print "C value chosen from: ", C_linear
print "Calculating accuracy with K-fold..."

for C in C_linear:
    svc_linear = svm.SVC(kernel='linear', C=C)
    scores = cross_validation.cross_val_score(svc_linear, X_train, y_train, scoring='accuracy', cv=6)
    result_linear.append(scores.mean())

print "result:", result_linear
#Result with different C are equal, so here choose C=1 directly as the best parameter.
best_param_linear = {"C": 3}


#linear_test_score = svm.SVC(kernel='linear', C=best_param_linear.get("C")).fit(X_test, y_test).score(X_test, y_test)
#rbf_test_score = svm.SVC(kernel='rbf', C=best_param_rbf.get("C"), gamma=best_param_rbf.get("gamma")).fit(X_test, y_test).score(X_test, y_test)
#poly_test_score = svm.SVC(kernel='poly', C=best_param_poly.get("C"), degree=best_param_poly.get("degree")).fit(X_test, y_test).score(X_test, y_test)
linear_test = svm.SVC(kernel='linear', C=best_param_linear.get("C")).fit(X, y)
count1 = 0
count2 = 0
for i in xrange(X_fin.__len__()):
    count2 += 1
    a = linear_test.predict(X_fin[i])
    b = y_fin[i]

    if a == [b]:
        count1 += 1

print "Total cases: ", count2
print "Correct Prediction: ", count1
print "Correct Rate: ", float(count1) / count2


#print "Linear Kernel test score: ", linear_test_score
#print "RBF Kernel test score: ", rbf_test_score
#print "Poly Kernel test score: ", poly_test_score
################################### Random Forests ####################################
print "##### Random Forest ######"
n_estimators_list = range(1, 16, 1)
result_random_forests = []
max_score_rf = float("-inf")
best_param_rf = None
for n_estimators in n_estimators_list:
    print "Testing n_estimators = ", n_estimators
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=1, random_state=0)
    scores = cross_validation.cross_val_score(rf_clf, X_train, y_train, scoring="accuracy", cv=6)
    result_random_forests.append(scores.mean())
    if scores.mean() > max_score_rf:
        max_score_rf = scores.mean()
        best_param_rf = {"n_estimators": n_estimators}
print "number of trees: ", n_estimators_list
print "results: ", result_random_forests
print "best accuracy: ", max_score_rf
print "best parameter: ", best_param_rf

rf_clf_test_score = RandomForestClassifier(n_estimators=best_param_rf.get("n_estimators"), max_depth=None,
                                           min_samples_split=1, random_state=0).fit(X_test, y_test).score(X_test,
                                                                                                          y_test)
print "Test set accuracy: ", rf_clf_test_score

rf_clf = RandomForestClassifier(n_estimators=best_param_rf.get("n_estimators"), max_depth=None, min_samples_split=1,
                                random_state=0).fit(X, y)
count1 = 0
count2 = 0
for i in xrange(X_fin.__len__()):
    count2 += 1
    a = rf_clf.predict(X_fin[i])
    b = y_fin[i]
    print "+ ", a[0],
    print "- ", b
    if a == [b]:
        count1 += 1

print "Total cases: ", count2
print "Correct Prediction: ", count1
print "Correct Rate: ", float(count1) / count2


################################### K Nearest Neighbors ####################################
print "##### K Nearest Neighbors ######"
n_neighbors_list = range(1, 6, 1)
result_n_neighbors = []
max_score_knn = float("-inf")
best_param_knn = None
for n_neighbors in n_neighbors_list:
    print "Testing n_neighbors = ", n_neighbors
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_validation.cross_val_score(neigh, X_train, y_train, scoring="accuracy", cv=6)
    result_n_neighbors.append(scores.mean())
    if scores.mean() > max_score_knn:
        max_score_knn = scores.mean()
        best_param_knn = {"n_neighbors": n_neighbors}
print "number of neighbors: ", n_neighbors_list
print "results: ", result_n_neighbors
print "best accuracy: ", max_score_knn
print "best parameter: ", best_param_knn

neigh_test_score = KNeighborsClassifier(best_param_knn.get("n_neighbors")).fit(X_test, y_test).score(X_test, y_test)
print "Test set accuracy: ", neigh_test_score

neigh = KNeighborsClassifier(best_param_knn.get("n_neighbors")).fit(X, y)
count1 = 0
count2 = 0
for i in xrange(X_fin.__len__()):
    count2 += 1
    a = neigh.predict(X_fin[i])
    b = y_fin[i]
    if a == [b]:
        count1 += 1

print "Total cases: ", count2
print "Correct Prediction: ", count1
print "Correct Rate: ", float(count1) / count2
