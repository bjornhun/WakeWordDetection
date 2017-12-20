import os
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from pandas import read_pickle

X_test, y_test = read_pickle("data/test.pickle")
(num_test, rows, columns) = X_test.shape

def evaluate_model(filename):
    global X_test

    if filename.startswith("mlp"):
        X_test = X_test.reshape(num_test, rows * columns)
    else:
        X_test = X_test.reshape(num_test, rows, columns, 1)

    model = load_model('models/' + filename)
    print("Test data:")
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(confusion_matrix(y_test, [int(round(x[0])) for x in model.predict(X_test)]))

if __name__ == '__main__':
    models = os.listdir('models')
    for model in models:
        print("\n\n...................\n")
        print(model + ":\n")
        evaluate_model(model)