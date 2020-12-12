import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
# The competition datafiles are in the directory ../input
# Read competition data files:
train_filename = 'UNSW_NB15_training-set.csv'
test_filename  = 'UNSW_NB15_testing-set.csv'

train = pd.read_csv(train_filename)
test = pd.read_csv(test_filename)

train_x = train.iloc[:,:-1]
train_y = train.iloc[:,-1]

test_x = test.iloc[:,:-1]
test_y = test.iloc[:,-1]



#print(train_x.shape,train_y.shape) # (82332, 44) (82332,)
print(train.columns)
'''
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


    
train_x = train_x.values
train_x = preprocess(train_x)
train_x = train_x.reshape((-1,42))

test_x = test_x.values
test_x = preprocess(test_x)
test_x = test_x.reshape((-1,42))

#print(train_x)
#print(train_y)

#pred_Y = clf.predict(test_X)
#num_class = len(unique)
#print('num class', num_class)
#print (np.asarray((unique, counts)).T)
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(train_x, train_y)


#Predict the response for test dataset
y_pred = clf.predict(test_x)
cm = confusion_matrix(test_y,y_pred)

print("Accuracy:",metrics.accuracy_score(test_y, y_pred))

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['benign','attack'], 
                     columns = ['benign','attack'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('SVM classifier\nAccuracy:{0:.3f}'.format(accuracy_score(test_y, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();
'''