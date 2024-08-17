from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
import pandas as pd
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

    return features_list,y

'''df = pd.read_csv("preprocessed_dataset.csv")
features_list,targetcol=getFeatures(df)
X=df[features_list]
y=targetcol
# Split train test: 80 % - 20 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)
dt_evaluation(X_train, X_test, y_train, y_test)'''