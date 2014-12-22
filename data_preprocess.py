__author__ = 'Chao'

activity_label = {'1': 'WALKING',
                  '2': 'WALKING_UPSTAIRS',
                  '3': 'WALKING_DOWNSTAIRS',
                  '4': 'SITTING',
                  '5': 'STANDING',
                  '6': 'LAYING'}

# ############################# 1 Open data set ###############################
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

try:
    open("train.csv", 'wt').close()
    train_csv = open("train.csv", 'a')
    open("test.csv", 'wt').close()
    test_csv = open("test.csv", 'a')
except:
    exit()
for i in xrange(y.__len__()):
    for j in xrange(X[i].__len__()):
        train_csv.write(str(X[i][j]))
        train_csv.write(',')
    train_csv.write(activity_label[str(y[i])])
    train_csv.write('\n')
for i in xrange(y_fin.__len__()):
    for j in xrange(X_fin[i].__len__()):
        test_csv.write(str(X_fin[i][j]))
        test_csv.write(',')
    test_csv.write(activity_label[str(y_fin[i])])
    test_csv.write('\n')

train_csv.close()
test_csv.close()
