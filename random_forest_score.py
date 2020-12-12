import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,recall_score, f1_score

from sklearn import metrics
#import tqdm

#data
train_filename='UNSW_NB15_training-set.csv'
test_filename='UNSW_NB15_testing-set.csv'

def read_csv(filename):
    with open(filename,'r') as dest_f:

        data_iter = csv.reader(dest_f,
                           delimiter = ',',
                           quotechar = '"')
        data = [data for data in data_iter]
    print(filename, 'length of the data read',len(data))
    return data
    
def set2dict(categories):
    categories = sorted(categories)
    counter=0
    cat_d = {}
    for c in categories:
        cat_d[c]=counter
        counter+=1
    return cat_d


def preprocess(data):
    data = np.array(data)
    data = np.delete(data,0,axis=0)
    proto_index = 2
    service_index = 3
    state_index = 4

    # process 
    proto = set(data[:,proto_index])
    service = set(data[:,service_index])
    state_proto = set(data[:,state_index])

    service_d = set2dict(service)
    state_proto_d = set2dict(state_proto)
    proto_d = set2dict(proto)

    for index,row in enumerate(data):
        data[index,service_index] = service_d[data[index,service_index]]
        data[index,state_index] = state_proto_d[data[index,state_index]]
        data[index,proto_index] = proto_d[data[index,proto_index]]

    data= np.delete(data,[0,43],axis=1)
    return data

#training
train_data = read_csv(train_filename)
test_data = read_csv(test_filename)

train_data = preprocess(train_data)
test_data = preprocess(test_data)
print('training data:',train_data.shape, 'test data:',test_data.shape)

train_Y = train_data[:,-1]
train_X = train_data[:,:-1]
unique, counts = np.unique(train_Y, return_counts=True)
#
clf = RandomForestClassifier()
clf.fit(train_X,train_Y)

test_Y = test_data[:,-1]
test_X = test_data[:,:-1]
#pred_Y = clf.predict(test_X)
#Predict the response for test dataset
y_pred = clf.predict(test_X)
cm = confusion_matrix(test_Y,y_pred)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(test_Y, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(test_Y, y_pred, average="macro")
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(test_Y, y_pred, average="macro")
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(test_Y, y_pred, average="macro")
print('F1 score: %f' % f1)
###############
# Transform to df for easier plotting
cm_df = pd.DataFrame(cmn,
                     index = ['benign','attack'], 
                     columns = ['benign','attack'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('RandomForestClassifier\nAccuracy:{0:.3f}'.format(accuracy_score(test_Y, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();

