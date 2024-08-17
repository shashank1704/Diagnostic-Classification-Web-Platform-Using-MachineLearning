

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def getFeatures(df):

    y = df["TenYearCHD"]  # target column i.e price range
    del df["TenYearCHD"]
    # separate independent & dependent variables
    X = df  # independent columns

    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    #print(featureScores.nlargest(11, 'Score'))  # print 10 best features
    featureScores = featureScores.sort_values(by='Score', ascending=False)
    print(featureScores)

    # selecting the 10 most impactful features for the target variable
    features_list = featureScores["Specs"].tolist()[:10]
    print(features_list)

    return df[features_list],y



df=pd.read_csv("dataset.csv")

df=df.dropna() # Remove the empty cell or NaN values of records

'''y=df["TenYearCHD"]

print(y)

del df["TenYearCHD"]

x=df

print(x)'''

x,y=getFeatures(df)


X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=30)


clf_knn=KNeighborsClassifier()

clf_knn.fit(X_train,y_train)

pre=clf_knn.predict(X_test)

print("KNN algorithm:")

acc_score=accuracy_score(y_test,pre)*100

pre_score=precision_score(y_test,pre)*100

rec_score=recall_score(y_test,pre)*100

f1score=f1_score(y_test,pre)*100

print(acc_score)

print(pre_score)

print(rec_score)

print(f1score)

'''testdata=[[0,63,2,1,40,0,0,0,0,179,116,69,22.15,95,75]]

pre_res=clf_knn.predict(testdata)

print("prediciton=",pre_res)

if pre_res[0]==0:
    print("Negative")
else:
    print("Positive")'''
