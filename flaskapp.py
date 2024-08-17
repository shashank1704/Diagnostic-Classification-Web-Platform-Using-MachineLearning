from flask import Flask,render_template,request
from FeatureSelection import getFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from DBConfig import DBConnection
import pandas as pd
import numpy as np
from RF import rfc_evaluation
from DT import dt_evaluation
from GB import gb_evaluation
from SVM import svm_evaluation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
from Similarity import similarity
from Image_Prediction import classify_image_cnn



accuracy_list=[]
accuracy_list.clear()
precision_list=[]
precision_list.clear()
recall_list=[]
recall_list.clear()
f1score_list=[]
f1score_list.clear()

app=Flask(__name__)
app.secret_key='abc'

dict={}
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/admin')
def admin():
    return render_template("admin.html")

@app.route('/admin_home')
def admin_home():
    return render_template("admin_home.html")

@app.route('/preprocessing')
def preprocessing():
    return render_template("data_preprocessing.html")

@app.route("/data_preprocessing" ,methods =["GET", "POST"] )
def data_preprocessing():
    fname = request.form.get("file")
    df = pd.read_csv(fname)
    df1 = df.dropna()
    df1.to_csv("preprocessed_dataset.csv",index=False)

    orecords=len(df)
    precords = len(df1)
    df1=df1[0:10]
    data=df1.values.tolist()
    print("data=",type(data))

    return render_template("data_preprocessing.html",rawdata=data,orecords=orecords,precords=precords)

@app.route("/features_selection" )
def features_selection():
    return render_template("features_selection.html")

@app.route("/selected_features",methods =["GET", "POST"] )
def selected_features():
    fname = request.form.get("file")
    df = pd.read_csv(fname)
    features_list,targetcol=getFeatures(df)
    X=df[features_list]
    y=targetcol
    dict['X'] = X
    dict['y'] = y


    return render_template("features_selection.html",features_list=features_list)

@app.route("/perevaluations")
def perevaluations():
    accuracy_graph()
    precision_graph()
    recall_graph()
    f1score_graph()
    return render_template("metrics.html")

def accuracy_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    accuracy_list.clear()

    cursor.execute("select accuracy from evaluations")
    acdata=cursor.fetchall()

    for record in acdata:
        accuracy_list.append(float(record[0]))

    height = accuracy_list

    bars = ('SVM','DT','RF','GB')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['red', 'green', 'blue', 'orange'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.title('Analysis on ML Accuracies')
    plt.savefig('static/accuracy.png')
    plt.clf()




def precision_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()

    cursor.execute("select precesion from evaluations")
    pdata = cursor.fetchall()

    precision_list.clear()
    for record in pdata:
        precision_list.append(float(record[0]))

    height = precision_list
    print("pheight=",height)
    bars = ('SVM','DT','RF','GB')
    y_pos = np.arange(len(bars))
    plt2.bar(y_pos, height, color=['green', 'brown', 'violet', 'blue'])
    plt2.xticks(y_pos, bars)
    plt2.xlabel('Algorithms')
    plt2.ylabel('Precision')
    plt2.title('Analysis on ML Precisions')
    plt2.savefig('static/precision.png')
    plt2.clf()


def recall_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    recall_list.clear()
    cursor.execute("select recall from evaluations")
    recdata = cursor.fetchall()

    for record in recdata:
        recall_list.append(float(record[0]))

    height = recall_list

    bars = ('SVM','DT','RF','GB')
    y_pos = np.arange(len(bars))
    plt3.bar(y_pos, height, color=['orange', 'cyan', 'gray', 'violet'])
    plt3.xticks(y_pos, bars)
    plt3.xlabel('Algorithms')
    plt3.ylabel('Recall')
    plt3.title('Analysis on ML Recall')
    plt3.savefig('static/recall.png')
    plt3.clf()


def f1score_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    f1score_list.clear()

    cursor.execute("select f1score from evaluations")
    fsdata = cursor.fetchall()

    for record in fsdata:
        f1score_list.append(float(record[0]))

    height = f1score_list

    bars = ('SVM','DT','RF','GB')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['gray', 'green', 'orange', 'brown'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('F1-Score')
    plt.title('Analysis on ML F1-Score')
    plt4.savefig('static/f1score.png')
    plt4.clf()


@app.route("/evaluations")
def evaluations():


    rf_list=[]
    dt_list = []

    svm_list = []
    gb_list = []

    metrics=[]
    X=dict['X']
    y=dict['y']

    # Split train test: 80 % - 20 %
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

    accuracy_svm, precision_svm, recall_svm, fscore_svm = svm_evaluation(X_train, X_test, y_train, y_test)

    svm_list.append("SVM")
    svm_list.append(accuracy_svm)
    svm_list.append(precision_svm)
    svm_list.append(recall_svm)
    svm_list.append(fscore_svm)

    accuracy_dt, precision_dt, recall_dt, fscore_dt = dt_evaluation(X_train, X_test, y_train, y_test)
    dt_list.append("DT")
    dt_list.append(accuracy_dt)
    dt_list.append(precision_dt)
    dt_list.append(recall_dt)
    dt_list.append(fscore_dt)

    accuracy_rf, precision_rf, recall_rf, fscore_rf = rfc_evaluation(X_train, X_test, y_train, y_test)
    rf_list.append("RF")
    rf_list.append(accuracy_rf)
    rf_list.append(precision_rf)
    rf_list.append(recall_rf)
    rf_list.append(fscore_rf)

    accuracy_gb, precision_gb, recall_gb, fscore_gb = gb_evaluation(X_train, X_test,y_train, y_test)
    gb_list.append("GB")
    gb_list.append(accuracy_gb)
    gb_list.append(precision_gb)
    gb_list.append(recall_gb)
    gb_list.append(fscore_gb)

    metrics.clear()
    metrics.append(svm_list)
    metrics.append(dt_list)
    metrics.append(rf_list)
    metrics.append(gb_list)


    return render_template("evaluations.html", evaluations=metrics)

@app.route("/adminlogin_check",methods =["GET", "POST"])
def adminlogin_check():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        if uid=="admin" and pwd=="admin":

            return render_template("admin_home.html")
        else:
            return render_template("admin.html",msg="Invalid Credentials")


@app.route("/user")

def user():
    return render_template("user.html")


@app.route("/newuser")
def newuser():
    return render_template("register.html")
@app.route("/user_register",methods =["GET", "POST"])
def user_register():
    try:
        sts=""
        name = request.form.get('name')
        uid = request.form.get('unm')
        pwd = request.form.get('pwd')
        mno = request.form.get('mno')
        email = request.form.get('email')
        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + uid + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            sts = 0
        else:
            sql = "insert into register values(%s,%s,%s,%s,%s)"
            values = (name,uid, pwd,email,mno)
            cursor.execute(sql, values)
            database.commit()
            sts = 1

        if sts==1:
            return render_template("user.html", msg="Registered Successfully..! Login Here.")


        else:
            return render_template("register.html", msg="User name already exists..!")



    except Exception as e:
        print(e)

    return ""

@app.route("/userlogin_check",methods =["GET", "POST"])
def userlogin_check():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")

        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + uid + "' and password='" + pwd + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:

            return render_template("user_home.html")
        else:

            return render_template("user.html", msg2="Invalid Credentials")

        return ""


@app.route("/hd_prediction")
def hd_prediction():
    return render_template("prediction.html")




@app.route("/prediction_hd", methods =["GET", "POST"])
def prediction_hd():

    df = pd.read_csv("preprocessed_dataset.csv")
    y_train=df['TenYearCHD']
    del df['TenYearCHD']

    features = ['sysBP', 'glucose', 'age', 'totChol', 'cigsPerDay', 'diaBP', 'prevalentHyp', 'diabetes', 'BPMeds',
                'male']
    X_train=df[features]

    age = request.form.get("age")
    gender = request.form.get("gender")
    cigpd = request.form.get("cigpd")
    sysbp = request.form.get("sysbp")
    diabp = request.form.get("diabp")
    chol = request.form.get("chol")
    prehyp = request.form.get("prehyp")
    diabetes = request.form.get("diabetes")
    glucose = request.form.get("glucose")
    bpm = request.form.get("bpm")
    X_test=[[sysbp,glucose,age,chol,cigpd,diabp,prehyp,diabetes,bpm,gender]]
    print(X_test)

    rfc_clf = RandomForestClassifier()
    rfc_clf.fit(X_train, y_train)

    predicted = rfc_clf.predict(X_test)
    result=predicted[0]
    print("res=",result)


    return render_template("prediction.html",result=result)



@app.route("/prediction_db", methods =["GET", "POST"])
def prediction_db():

    df = pd.read_csv("diabetes_prediction_dataset.csv")

    df.dropna()

    # converting necessary columns to numerical type
    from sklearn.preprocessing import LabelEncoder
    label = LabelEncoder()
    df['gender'] = label.fit_transform(df['gender'])




    y_train=df['diabetes']
    del df['smoking_history']
    del df['diabetes']


    X_train=df

    age = request.form.get("age")
    gender = request.form.get("gender")
    hyps = request.form.get("hyps")
    hd = request.form.get("hd")
    bmi = request.form.get("bmi")
    hba = request.form.get("hba")
    glucose = request.form.get("glucose")


    X_test=[[age,gender,hd,hyps,bmi,hba,glucose]]
    print(X_test)

    rfc_clf = RandomForestClassifier()
    rfc_clf.fit(X_train, y_train)

    predicted = rfc_clf.predict(X_test)
    result=predicted[0]
    print("res=",result)


    return render_template("prediction_db.html",result=result)


@app.route("/db_prediction")
def db_prediction():
    return render_template("prediction_db.html")

@app.route("/spam_prediction")
def spam_prediction():
    return render_template("spam_prediction.html")

@app.route("/prediction_spam", methods =["GET", "POST"])
def prediction_spam():
    from sklearn.feature_extraction.text import TfidfVectorizer

    from sklearn.ensemble import RandomForestClassifier

    msg = request.form.get("msg")

    df = pd.read_csv('spam_train.csv')

    x_train = df['Emails']

    y_train = df['Class']

    tfidf = TfidfVectorizer(stop_words='english', use_idf=False, smooth_idf=False)  # TF-IDF

    x_train = tfidf.fit_transform(x_train)

    x_test = tfidf.transform([msg])

    clf_rf = RandomForestClassifier()

    clf_rf.fit(x_train, y_train)

    pre = clf_rf.predict(x_test)

    result=pre[0]

    print(result)

    return render_template("spam_prediction.html", result=result)


@app.route("/chat")
def chat():
    database = DBConnection.getConnection()
    cursor = database.cursor()
    cursor2 = database.cursor()
    cursor.execute("delete from msgs")
    database.commit()

    '''cursor.execute("insert into msgs(msg,user_,time_) values('How may I help you?','chatbot',now())")
    database.commit()

    cursor.execute("insert into msgs(msg,user_,time_) values('"+msg+"','chatbot',now())")
    database.commit()'''
    cursor.execute("insert into msgs(msg,user_,time_) values('How can I help you..', 'chatbot',now())")
    database.commit()

    sql = "select * from msgs order by sno "
    cursor2.execute(sql)
    records = cursor2.fetchall()

    return render_template("chatbot.html", rawdata=records)

@app.route("/send",methods =["GET", "POST"])
def send():
    dict={}
    uid = "user"
    text = request.form.get('text')
    database = DBConnection.getConnection()
    cursor = database.cursor()
    cursor4 = database.cursor()
    cursor5 = database.cursor()
    cursor.execute("insert into msgs(msg,user_,time_) values('"+text+"','"+uid+"',now())")
    database.commit()

    cursor2 = database.cursor()
    sql = "select keywords,sno from questions"
    cursor2.execute(sql)
    res = cursor2.fetchall()

    for row in res:
        score=similarity(row[0], text)
        dict[row[1]]=score

    key = max(dict, key=dict.get)
    print("Highest value from dictionary:", key)

    if  dict[key]>0.1:
        cursor3 = database.cursor()
        sql2 = "select reply from questions where sno='"+str(key)+"' "
        cursor3.execute(sql2)
        res2 = cursor3.fetchall()

        for row2 in res2:

            replay=row2[0]

            cursor4.execute("insert into msgs(msg,user_,time_) values('" + replay + "','chatbot',now())")
            database.commit()

        sql3 = "select * from msgs order by sno "
        cursor5.execute(sql3)
        records = cursor5.fetchall()

        return render_template("chatbot.html", rawdata=records)




    else:
        cursor4.execute("insert into msgs(msg,user_,time_) values('Sorry, I am not understood..','chatbot',now())")
        database.commit()

        sql3 = "select * from msgs order by sno "
        cursor5.execute(sql3)
        records = cursor5.fetchall()

        return render_template("chatbot.html", rawdata=records)

    return render_template("")

@app.route("/image_classification")
def image_classification():
    return render_template("prediction_image.html")


@app.route("/prediction_image",methods =["GET", "POST"])
def prediction_image():
    try:

        image = request.files['file']

        image.save("../MLProject/test_image/test_img.jpg")

        image_path="../MLProject/test_image/test_img.jpg"

        result=classify_image_cnn(image_path)


    except Exception as e:
        print(e)

    return render_template("results.html", result=result)


@app.route("/face_detection",methods =["GET", "POST"])
def face_detection():
    import cv2
    import numpy as np

    cam = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    while True:
        ret, img = cam.read()

        img_copy = np.copy(img)
        # convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1);

        # go over list of faces and draw them as rectangles on original colored img
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('FaceDetection', img_copy)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        print("\n close camera")
    cam.release()
    cv2.destroyAllWindows()







if __name__=="__main__":
    app.run(host="localhost",port="2024",debug=True)