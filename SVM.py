
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from DBConfig import DBConnection
def svm_evaluation(X_train, X_test, y_train, y_test):
    db = DBConnection.getConnection()
    cursor = db.cursor()
    cursor.execute("delete from evaluations")
    db.commit()
    svm_clf = SVC()

    svm_clf.fit(X_train, y_train)

    predicted = svm_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)*100

    precision = precision_score(y_test, predicted, average="macro")*100

    recall = recall_score(y_test, predicted, average="macro")*100

    fscore = f1_score(y_test, predicted, average="macro")*100

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=predicted)

    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix of SVM', fontsize=18)
    plt.savefig('static/cm_svm.png')
    plt.clf()

    values = ("SVM",  str(accuracy), str(precision), str(recall), str(fscore))
    sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()

    print("SVM=",accuracy,precision,recall,fscore)

    return accuracy, precision, recall, fscore




