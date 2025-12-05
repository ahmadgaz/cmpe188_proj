import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv("kickstarter_data_full copy.csv", encoding="latin-1", on_bad_lines="skip")
print(df.head())
print(df.info())

df['usd_goal'] = df['goal'] * df['static_usd_rate']

df.drop([
    'Unnamed: 0','deadline','created_at','launched_at','id','photo','slug', 'friends',
    'is_starred','is_backing','permissions', 'urls', 'source_url','profile','creator','state',
    "state_changed_at","state_changed_at_weekday","state_changed_at_month","state_changed_at_day",
    "state_changed_at_yr","state_changed_at_hr","launch_to_state_change",
    "launch_to_state_change_days", 'currency_symbol','currency_trailing_code', 'goal',
    'static_usd_rate', 'location', "disable_communication","spotlight","staff_pick",
    "pledged","usd_pledged","backers_count", 'create_to_launch','launch_to_deadline'
], axis=1, inplace=True)

df.dropna(inplace=True)
print(df.head())

cat_cols = [
    "country","currency","category","deadline_weekday","created_at_weekday",
    "launched_at_weekday","deadline_month","created_at_month","launched_at_month",
    "deadline_day","created_at_day","launched_at_day","deadline_hr","created_at_hr",
    "launched_at_hr","USorGB","TOPCOUNTRY","LaunchedTuesday","DeadlineWeekend"
]

text_cols = ['name','blurb']

X = df.drop(['SuccessfulBool'] + text_cols, axis=1)
y = df['SuccessfulBool']

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

K_BEST = 35

selector = SelectKBest(score_func=f_classif, k=K_BEST)
selector.fit(X_train, y_train)

mask = selector.get_support()
selected_features = X_train.columns[mask]
print("Selected features:")
print(selected_features)


X_train_sel = X_train[selected_features]
X_test_sel  = X_test[selected_features]

from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

BEST = 13


selector = SelectKBest(score_func=f_classif, k=K_BEST)
selector.fit(X_train, y_train)

X_train_selected = selector.transform(X_train)
X_test_selected  = selector.transform(X_test)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_selected,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

selector = SelectKBest(score_func=f_classif, k=K_BEST)
selector.fit(X_train, y_train)

svm_model = SVC(
    kernel='linear',  #alter this
    C=1,              #0.1-1 seems best
    probability=True,
    random_state=42
)

svm_model.fit(X_train_selected, y_train)

y_pred = svm_model.predict(X_test_selected)
y_prob = svm_model.predict_proba(X_test_selected)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("Accuracy:", acc)
print("ROC:", roc)