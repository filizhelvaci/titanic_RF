import pandas as pd
import pickle
import warnings
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from helpers.data_prep import *
from helpers.eda import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

data_train = pd.read_csv("dataset/train.csv")
data_tr=data_train.drop(["PassengerId"],axis=1)

data_test=pd.read_csv("dataset/test.csv")
data_t=data_test.drop(["PassengerId"],axis=1)
data_t.head()

data_ts=pd.read_csv("hafta8/gender_submission.csv")
data_ts.head()
data_t_y=data_ts.drop(["PassengerId"], axis=1)

def titanic_data_prep(df):

    ### DATA ENGİNEER

    df["NEW_CABIN_BOOL"] = df["Cabin"].isnull().astype("int")
    df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"  # Gemide yalnızmı? Hayır
    df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"  # Gemide yalnız mı Evet
    df["NEW_NAME_COUNT"] = df["Name"].str.len()
    df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
    df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    df["NEW_TITLE"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)
    df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

    df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    df.loc[(df["Age"] < 18), "NEW_AGE_CAT"] = "young"
    df.loc[(df["Age"] >= 18) & (df["Age"] < 56), "NEW_AGE_CAT"] = "mature"
    df.loc[(df["Age"] >= 56), "NEW_AGE_CAT"] = "senior"

    df.columns = [col.upper() for col in df.columns]

    ### AYKIRI DEGERLER

    num_cols = [col for col in df.columns if len(df[col].unique()) > 20
                and df[col].dtypes != 'O'
                and col not in "PASSENGERID"]

    for col in num_cols:
        replace_with_thresholds(df, col)

    # for col in num_cols:
    #    print(col, check_outlier(df, col))

    # check_df(df)

    ### EKSİK DEGERLER

    #missing_values_table(df, na_name=True)
    df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
    df = df.apply(lambda x: x.fillna(x.mode()[0]) , axis=0)
    df.drop(["TICKET", "NAME", "CABIN"], inplace=True, axis=1)

    ### LABEL ENCODER

    binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

    for col in binary_cols:
        df = label_encoder(df, col)

    ### RARE ENCODER

    df = rare_encoder(df, 0.01)

    ### ONE HOT ENCODER

    ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
    df = one_hot_encoder(df, ohe_cols)

    return df

# Train dataseti veri önişleme yapıyoruz
df=titanic_data_prep(data_tr)
check_df(df)
# Test dataseti veri önişleme yapıyoruz
dff=titanic_data_prep(data_t)
check_df(dff)

##############################################################################
# Train.csv ile Random Forest Model oluşturma
###############################################################################
# Random Forests
y=df["SURVIVED"]
X=df.drop(["SURVIVED"],axis=1)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=17)

# Random Forests : Model & Tahmin
rf_model=RandomForestClassifier(random_state=42).fit(X_train,y_train)

# İlkel test hatamız
y_pred=rf_model.predict(X_test)
accuracy_score(y_test,y_pred) # 0.76

# Model Tuning

rf=RandomForestClassifier()
rf_params={"n_estimators":[600,750],
           "max_features":[5,10,12],
           "min_samples_split":[8,10,12]}
rf_cv_model=GridSearchCV(rf,rf_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
rf_cv_model.best_params_ # {'max_features': 5, 'min_samples_split': 10, 'n_estimators': 600}
# {'max_features': 12, 'min_samples_split': 12, 'n_estimators': 750}

# final model
rf_tuned=RandomForestClassifier(max_features=5,min_samples_split=10,n_estimators=600).fit(X_train,y_train)
y_pred=rf_tuned.predict(X_test)
accuracy_score(y_test,y_pred) # 0.7947761194029851

rf_tuned
rf_tuned.feature_importances_

# değişken önem düzeyleri
feature_imp=pd.Series(rf_tuned.feature_importances_,index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp,y=feature_imp.index,capsize=10)
plt.xlabel("Değişken Önem Skorları")
plt.ylabel("değişkenler")
plt.title("Değişken Önem Düzeyleri")
plt.show()

#################################################################
# Test setiyle modeli test edicez
#################################################################
# X ve y değerlerini tüm train verisiyle yapıyoruz(X_train,y_train,X_test,y_test diye ayırmıyoruz)
y=df["SURVIVED"]
X=df.drop(["SURVIVED"],axis=1)

# final model (daha önceki parametreleri kullanarak modeli train veri setiyle eğitiyoruz)
rf_tuned=RandomForestClassifier(max_features=5,min_samples_split=10,n_estimators=600).fit(X,y)
y_pred=rf_tuned.predict(dff) # (test verisini(dff) kullanarak y_predleri oluşruyoruz)
accuracy_score(data_t_y,y_pred) # 0.9043062200956937 (gerçek test seti bağımlı değişkeni ile y_predleri karşılaştırıyoruz. tahmin başarımızı buluyoruz.)

# Kaggle'a göndermek için y_pred değerlerini  passengerId ile merge edip csv ile kaydedip göndericz
data_ttnc=data_ts.drop(["Survived"], axis=1)
data_ttnc["Survived"]=np.round_(y_pred)
data_ttnc.head()
data_ttnc.to_csv("DataSet/dataframe_t.csv",index=False)

# değişken önem düzeyleri
rf_tuned.feature_importances_

feature_imp=pd.Series(rf_tuned.feature_importances_,index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp,y=feature_imp.index,capsize=10)
plt.xlabel("Değişken Önem Skorları")
plt.ylabel("değişkenler")
plt.title("Değişken Önem Düzeyleri")
plt.show()