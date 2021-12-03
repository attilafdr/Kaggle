"""
Generates a submission file for the Kaggle Titanic challenge
"""

import numpy as np
import pandas as pd

from sklearn import svm, linear_model, ensemble, tree
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def data_preprocess(df):
    # Transform the categorical sex value to one-hot representation, so it can be processed independently
    df_sex = pd.get_dummies(data=df['Sex'], columns=['Sex'], drop_first=False)
    df_embarked = pd.get_dummies(data=df['Embarked'], columns=['Embarked'], drop_first=False)

    # Pclass, Age, Fare,

    np_data = np.array(pd.concat([df['Age'], df['Pclass'], df['Fare'], df_embarked, df_sex], axis=1))
    np_ids = np.array(df['PassengerId'])

    # Fill missing values
    data_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    np_data = data_imputer.fit_transform(np_data)

    if 'Survived' in df.columns:
        labels_arr = np.array(df['Survived'])
    else:
        labels_arr = None

    return np_data, np_ids, labels_arr

if __name__ == '__main__':
    # Load all available training data
    df_all_train = pd.read_csv('./titanic/data/train.csv')

    # Generate 5 data splits to evaluate training method
    kf_splits = KFold(n_splits=5, shuffle=True, random_state=2)

    # Track run metrics for aggregation later
    df_scores = pd.DataFrame(columns=['F1', 'Precision', 'Recall', 'Accuracy'])
    df_model_agg_scores = pd.DataFrame(columns=['Model', 'Avg score', 'Min score', 'Max score'])

    # Models to be trained and compared
    models = [linear_model.RidgeClassifier(alpha=.5),
              tree.DecisionTreeClassifier(),
              linear_model.LogisticRegression(solver='liblinear', random_state=42),
              ensemble.GradientBoostingClassifier(),
              ensemble.RandomForestClassifier(),
              linear_model.SGDClassifier(loss="log"),
              svm.SVC()]

    # Run an experiment for each split and log the scores
    for model in models:
        for k, kf_split in enumerate(kf_splits.split(df_all_train)):
            df_train = df_all_train.iloc[kf_split[0]]
            df_test = df_all_train.iloc[kf_split[1]]

            train_arr, _, train_labels_arr = data_preprocess(df_train)

            model.fit(train_arr, train_labels_arr)

            test_arr, test_ids_arr, test_labels_arr = data_preprocess(df_test)
            pred_arr = model.predict(test_arr)

            scores = {'F1': f1_score(test_labels_arr, pred_arr),
                      'Precision': precision_score(test_labels_arr, pred_arr),
                      'Recall': recall_score(test_labels_arr, pred_arr),
                      'Accuracy': accuracy_score(test_labels_arr, pred_arr)}

            df_scores = df_scores.append(scores, ignore_index=True)

        # Aggregare and log results
        df_mean = df_scores.aggregate(['mean', 'min', 'max'], axis=0)
        df_model_agg_scores = df_model_agg_scores.append({'Model': str(model.__class__).split('.')[-1][:-2],
                                                          'Avg score': df_mean.iloc[0]['F1'],
                                                          'Min score': df_mean.iloc[1]['F1'],
                                                          'Max score': df_mean.iloc[2]['F1']},
                                                         ignore_index=True)

    print(df_model_agg_scores)

    # Find best model
    model = models[df_model_agg_scores['Avg score'].idxmax()]

    # Fit final model with all data
    train_arr, _, train_labels_arr = data_preprocess(df_all_train)
    model.fit(train_arr, train_labels_arr)

    # Load and prepare test data
    df_test = pd.read_csv('./titanic/data/test.csv')
    test_arr, test_ids_arr, _ = data_preprocess(df_test)

    # Make predictions
    pred_arr = model.predict(test_arr)

    # Construct output data format
    out = np.stack([test_ids_arr, pred_arr]).transpose()
    np.savetxt("output.csv", out, delimiter=",", fmt='%d', header='PassengerId,Survived')

