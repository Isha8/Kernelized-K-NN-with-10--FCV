import csv
import sys
import pandas as pd
from pandas import DataFrame
import math
import numpy
import operator
from sklearn.cross_validation import KFold
from sklearn import preprocessing
pd.__version__

def distanceMeasure(instance1, instance2, length):
    distance = 0
    distance_Polynomial = 0
    for x in range(length):
        distance_Polynomial += (instance1[x]-instance2[x])
        distance += pow((instance1[x] - instance2[x]), 2)
        if sys.argv[2]=="a":
            return math.sqrt(distance)
        if sys.argv[2] == "b":
            return math.pow((1+ math.sqrt(distance)),float(p))
        if sys.argv[2]=="c":
            sigmaSq=math.pow(float(sigma),2)
            return math.exp(-abs(instance1[x]-instance2[x])/float(sigma))

def find_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = distanceMeasure(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(3):
        neighbors.append(distances[x][0])
    return neighbors

def get_response(neighbors):
    class_found = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_found:
            class_found[response] += 1
        else:
            class_found[response] = 1
    sorted_classes_found = sorted(class_found.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_classes_found[0][0]


def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

"""file_name = raw_input("Enter the file name(File must be stored in the same folder as the executable)")"""

with open(sys.argv[3], 'rU') as csvfile:
    file_read = csv.reader(csvfile)
    heading = next(file_read)
    no_of_columns = len(next(file_read))
    no_of_rows = 1
    print (heading)
    for row in file_read:
       print (row)
       no_of_rows= no_of_rows+1

print ("\n Attributes \n" + str(heading))

print ("\nNo of examples in file ="+ str(no_of_rows))

col_lis = []
for i in range(0,no_of_columns-1,1):
    col_lis.append(i)

#print(col_lis);


def distanceMeasure(instance1, instance2, length):
    distance = 0
    distance_Polynomial = 0
    for x in range(length):
        distance_Polynomial += (instance1[x]-instance2[x])
        distance += pow((instance1[x] - instance2[x]), 2)
    if sys.argv[2]=="a":
        return math.sqrt(distance)
    if sys.argv[2] == "b":
        return math.pow((1+ math.sqrt(distance)),float(p))
    if sys.argv[2]=="c":
        sigmaSq=math.pow(float(sigma),2)
        return math.exp(-(distance_Polynomial)/float(sigma))

"""def find_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = distanceMeasure(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(3):
        neighbors.append(distances[x][0])
    return neighbors """

def get_response(neighbors):
    class_found = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_found:
            class_found[response] += 1
        else:
            class_found[response] = 1
    sorted_classes_found = sorted(class_found.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_classes_found[0][0]


def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0




"""**************Storing and Printing Examples***********************"""
data_X= pd.read_csv(sys.argv[3],nrows=no_of_rows,usecols=col_lis)
print ("\n")
print (data_X)

#df= pd.DataFrame(data_X)
#print "Converted to DataFrame"
#print df
"""**************Storing and Printing Class Column********************"""

f_X = pd.read_csv(sys.argv[3], usecols=[no_of_columns-1])
print ("\n")
print( f_X)


"""***********Data Normalization***********"""
column={}
max_column={}
min_column={}
for i in range(0,no_of_columns-1,1):
    column[i] = pd.read_csv(sys.argv[3], usecols=[i],nrows=no_of_rows )
    max_column[i] = float(column[i].max())
    min_column[i] = float(column[i].min())
#print max_column
#print "\n"
#print min_column


normalized_data=preprocessing.normalize(data_X)
#print normalized_data

#data=[[]]
#for i in range(0,no_of_columns-1,1):
 #   for j in range(0, no_of_rows - 1, 1):
  #      data[j][i] = pd.read_csv("ecoli.csv", usecols=[i], )

normalized_withoutClass= pd.DataFrame(normalized_data)
normalized_df = pd.DataFrame.merge(normalized_withoutClass,f_X,left_index= True, right_index= True)
"""normalized_df = normalized_df_withoutClass.append(f_X)"""
#print normalized_df

"""""""*******************K-fold Cross Validation*******************"""
no_of_folds= input('Enter the parameter k for k-Fold:')

"""
kf=KFold(len(data_X),n_folds=int(k),shuffle=True,random_state=123456)
print kf


train_data={}
train_data=heading
test_data={}
test_data =heading

count = 1
for i, j in kf:
    #train=X.iloc[i], test=X.iloc[j]
    #print train,test
    train = i
    for a in list(train):
        train_data=normalized_df.ix[[a]]
    print "train" + str(train_data)
    test = j
    train_data=heading
    for a in list(test):
        test_data=normalized_df.ix[[a]]
    print "test"+ str(test_data)
    test_data = heading
"""

train_final={}
test_final={}
ex = KFold(len(data_X), n_folds=int(no_of_folds),shuffle=True)
print ("\n")
"""k = raw_input('Enter the parameter k for kNN:')"""
accuracy=0;
if sys.argv[2]== "c":
    sigma = input("Enter the value of sigma for Radial Basis Kernel:");
if sys.argv[2]== "b":
    p = input("Enter the value of degree for Polynomial Kernel:");
for train_ex, test_ex in ex:
 #print("%s %s" % (train_ex, test_ex))
 train_final=normalized_df.ix[train_ex]
 print ("Training Data:")
 print (train_final)
 print ("\n")
 test_final = normalized_df.ix[test_ex]
 print ("Testing Data:")
 print (test_final)
 print ("\n")
 train = train_final.values.tolist();

 test = test_final.values.tolist();
 predictions = []

 for x in range(len(test_final)):
         nearest_neighbors = find_neighbors(train, test[x], sys.argv[1])
         result = get_response(nearest_neighbors)
         predictions.append(result)
         print('-> Predicted Class=' + repr(result) + ' and Actual Class=' + repr(test[x][-1]))
 accuracy = accuracy + get_accuracy(test, predictions)

final_accuracy = accuracy/int(no_of_folds);
print('Accuracy: ' + repr(final_accuracy) + '%')
























