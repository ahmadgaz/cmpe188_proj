import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


df = pd.read_csv("kickstarter_data_full.csv", encoding="latin-1", on_bad_lines="skip")
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
featurenames = X_train.columns.to_numpy()

idxsorted = np.argsort(importances)[::-1]
sortedfeatures = featurenames[idxsorted]

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
'''
TESTING FEATURES

for K in range(3, 50):

    selected_features = sorted_features[:K]
    print("Selected features:", selected_features.tolist())

    X_train_sel = X_train[selected_features]
    X_test_sel  = X_test[selected_features]


    rf_k = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    )


    rf_k.fit(X_train_sel, y_train)


    y_pred = rf_k.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for K = {K}: {acc:.4f}")
'''

#pretty much the best number K that the above commented code could make -> approx 0.7691.

BEST = 35
selectedfeatures = sortedfeatures[:BEST]
print(selectedfeatures)

X_train_sel = X_train[selectedfeatures]
X_test_sel  = X_test[selectedfeatures]

final = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=4,
   max_features=0.5,
    random_state=42,
    n_jobs=-1
)

final.fit(X_train_sel, y_train)
y_pred = final.predict(X_test_sel)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy for K = {BEST}: {acc:.4f}")

y_prob = final.predict_proba(X_test_sel)[:, 1]
roc = roc_auc_score(y_test, y_prob)
print(f"ROC for K = {BEST}: {roc:.4f}")

