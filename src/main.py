import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from joblib import dump
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.utils import preprocessing


def main():
    X, Y = preprocessing(dataframe=pd.read_csv("../raw/dataset.csv"),
                         selected_columns=['Product', 'Close_Value', 'Created Date', 'Close Date', 'Stage'],
                         missing_index=[1],
                         encode_index=[0],
                         need_extract_labels=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

    sc = StandardScaler()
    X_train[:, 7:9] = sc.fit_transform(X_train[:, 7:9])
    X_test[:, 7:9] = sc.transform(X_test[:, 7:9])

    classifierKnn = KNN(X_train, Y_train)
    classifierDt = DT(X_train, Y_train)
    classifierRf = RF(X_train, Y_train)
    classifierNb = NB(X_train, Y_train)

    dump(classifierDt, 'best_model.joblib')

    execute(classifierKnn, X_test, Y_test, "KNN")
    execute(classifierDt, X_test, Y_test, "Decision Tree")
    execute(classifierRf, X_test, Y_test, "Random Forest")
    execute(classifierNb, X_test, Y_test, "Naive Bayes")
    plt.show()


def execute(classifier, X_test, Y_test, algorithm_name):
    Y_pred = classifier.predict(X_test)
    print(f"{algorithm_name}:")
    print(f"accuracy= {accuracy_score(Y_test, Y_pred)}")
    print(f"F1= {f1_score(Y_test, Y_pred)}")
    print("-------------------------------")
    disp = plot_confusion_matrix(classifier, X_test, Y_test, cmap=plt.cm.Blues)
    disp.figure_.text(.5, .01, algorithm_name)


def KNN(X_train, Y_train):
    classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    classifier.fit(X_train, Y_train)
    return classifier


def DT(X_train, Y_train):
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, Y_train)
    return classifier


def RF(X_train, Y_train):
    classifier = RandomForestClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, Y_train)
    return classifier


def NB(X_train, Y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    return classifier


if __name__ == '__main__':
    main()
