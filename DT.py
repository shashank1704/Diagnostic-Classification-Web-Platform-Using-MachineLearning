
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from DBConfig import DBConnection

def dt_evaluation(X_train, X_test, y_train, y_test):
    db = DBConnection.getConnection()
    cursor = db.cursor()

    dtc_clf = DecisionTreeClassifier(criterion='entropy',max_depth=6)

    dtc_clf.fit(X_train, y_train)

    predicted = dtc_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)*100

    precision = precision_score(y_test, predicted, average="macro")*100

    recall = recall_score(y_test, predicted, average="macro")*100

    fscore = f1_score(y_test, predicted, average="macro")*100

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=predicted)

    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix of DT', fontsize=18)
    plt.savefig('static/cm_dt.png')
    plt.clf()


    values = ("DT", str(accuracy), str(precision), str(recall), str(fscore))
    sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()

    print("DTC=",accuracy,precision,recall,fscore)
    return accuracy,precision,recall,fscore





