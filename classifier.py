from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torch import nn

class Svm():
    def __init__(self, kernel='rbf', C=0.1):
        self.model = svm.SVC(kernel=kernel, C=C, decision_function_shape='ovr',class_weight='balanced')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
def use_svm():
    fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapping = torch.load(fpath + 'EEG_data/chunks/mapping.pt', map_location=torch.device('cpu'))
    totBatches = mapping.shape[0]
    X = np.arange(0, totBatches)
    tr_idx, test_idx = train_test_split(X,shuffle=True,test_size=0.2)
    currSub = "sub-01"
    eegdata = torch.load(fpath + '/EEG_data/chunks/' + currSub + ".pt",map_location=device)
    print(eegdata.shape)
    eegdata = torch.flatten(eegdata,start_dim=0,end_dim=1)
    eegdata_np = eegdata.numpy()
    eegdata_np = eegdata_np.reshape(eegdata_np.shape[0], -1)
    print(eegdata_np.shape)
    print(mapping.shape)
    labels = np.empty(mapping.shape[0]*mapping.shape[1])
    i = 0
    for row in mapping:
        for col in row:
            for index in range(len(col)):
                if col[index] == 1:
                    labels[i] = index
            i += 1
    labels = np.array(labels)
    print(labels.shape)

    #eegdata_np = eegdata_np[:1000]
    #labels = labels[:1000]

    print(eegdata_np.shape)
    print(labels.shape)
    

     # Replace this
    X_train, X_test, y_train, y_test = train_test_split(eegdata_np, labels, test_size=0.2, random_state=42)

    svm_classifier = Svm()

    svm_classifier.train(X_train, y_train)

    accuracy = svm_classifier.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    use_svm()