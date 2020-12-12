import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
# The competition datafiles are in the directory ../input
# Read competition data files:
train_filename = 'UNSW_NB15_training-set_balanced.csv'# use balanced for pca 
test_filename  = 'UNSW_NB15_testing-set_balanced.csv'

train = pd.read_csv(train_filename)
test = pd.read_csv(test_filename)

train_x = train.iloc[:,:-1]
train_y = train.iloc[:,-1]

test_x = test.iloc[:,:-1]
test_y = test.iloc[:,-1]


print(train_x.shape,train_y.shape) # (238682, 44) (238682,)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten,Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D
from keras.constraints import maxnorm
from keras.optimizers import adam
from keras.optimizers import SGD
from matplotlib import pyplot
from keras import backend as K
K.set_image_dim_ordering('th')

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
	scaler = MinMaxScaler()
	normalized = scaler.fit_transform(data)
	inverse = scaler.inverse_transform(normalized)
	###
	return data


def onehot(labels):
	Uniques,Index  = np.unique(labels,return_inverse=True)
	return np_utils.to_categorical(Index,len(Uniques))
	
def inverse_onehot(matrix):
	labels =[]
	for row in matrix:
		labels.append(np.argmax(row,axis=0))
	return labels    
	
def fit_pca(x):
	old_dim = x.shape[1]  # 42 feature in oure case
	new_dim = 35 #new feature
	
	pca = PCA(n_components=new_dim)
	new_data = pca.fit_transform(x) 
	# new_data.shape == 35
	return new_data


labels_train = onehot(train_y.values)
train_x = train_x.values # train_x.shape = (N,42)
train_x = preprocess(train_x)
train_x = fit_pca(train_x)
train_x = train_x.reshape((-1,35,1))

labels_test = onehot(test_y.values)
test_x = test_x.values
test_x = preprocess(test_x)
test_x = fit_pca(test_x)
test_x = test_x.reshape((-1,35,1))


model = Sequential()
model.add(Convolution1D(32,5,input_shape=(35,1),activation='sigmoid',border_mode='same' ,W_constraint=maxnorm(3)))
#32 filters, 5 kernal size, 35 features, 1 time step.

model.add(MaxPooling1D(pool_size=(2),strides=(1)))
#model.add(Convolution1D(32,5,activation='sigmoid',border_mode='same' ,W_constraint=maxnorm(3)))
#model.add(Convolution1D(32,5,activation='sigmoid',border_mode='same' ,W_constraint=maxnorm(3)))
#model.add(MaxPooling1D(pool_size=(2),strides=(1)))
#model.add(Convolution1D(32,5,activation='sigmoid',border_mode='same' ,W_constraint=maxnorm(3)))
#model.add(MaxPooling1D(pool_size=(2),strides=(1)))
model.add(Flatten())
model.add(Dense(32,activation='relu',W_constraint=maxnorm(3)))# batch normalizatin 32
model.add(Dropout(0.4))
model.add(Dense(labels_train.shape[1],activation='softmax'))#fully connnected layer
model.compile(optimizer=adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history=model.fit(train_x,#features
				  labels_train,#targets
				  epochs=200,#number of epochs
				  verbose=0, # No output
				  batch_size=32,# number of observations per batch
				  validation_data=(test_x, labels_test)) #data for evaluation
###############
_,train_acc = model.evaluate(train_x, labels_train, verbose=0)
_,test_acc = model.evaluate(test_x, labels_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc )) 

training_accuracy = history.history['acc']
test_accuracy = history.history['val_acc']

# Create count of the number of epochs
epoch_count = range(1, len(training_accuracy) + 1)
##########
# predict probabilities for test set
yhat_probs = model.predict(test_x, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(test_x, verbose=0)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(test_y, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(test_y, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(test_y, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(test_y, yhat_classes)
print('F1 score: %f' % f1)

# confusion matrix
matrix = confusion_matrix(test_y, yhat_classes)
# Normalise
cmn = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Transform to df for easier plotting
cm_df = pd.DataFrame(cmn,
                     index = ['normal','attack'], 
                     columns = ['normal','attack'])
# Visualize confusion matrix
plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('1D-CNN classifier\nAccuracy:{0:.3f}'.format(accuracy_score(test_y, yhat_classes)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
# Visualize accuracy history
plt.figure(figsize=(5.5,4))
plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy Score')
plt.show()

############
