from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Svm:
    def __init__(self, kernel='linear', C=1.0):
        self.model = svm.SVC(kernel=kernel, C=C, decision_function_shape='ovr')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
def use_svm():
     # Replace this
    X_train = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    y_train = [0, 0, 1, 1, 2, 2]
    X_test = [[0.2, 0.2], [0.8, 0.8], [2.2, 2.2], [2.8, 2.8], [4.2, 4], [5, 6]]
    y_test = [0, 0, 1, 1, 2, 2]
    # X_train, X_test, y_train, y_test = todo

    svm_classifier = Svm()

    svm_classifier.train(X_train, y_train)

    accuracy = svm_classifier.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    use_svm()