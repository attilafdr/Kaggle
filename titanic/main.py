"""
Selects the best model and generates a submission file for the Kaggle Titanic challenge
"""

import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

class DataPreprocessor(object):
    """Some preprocessing parameters are learned from the data"""
    def __init__(self, data, cols):
        self.surv_lookups = {}
        self.avg_surv_rate = 0

        self.generate_replace_lookups(data, cols)

    def generate_replace_lookups(self, data, cols):
        """Learn preprocessing parameters"""
        # Calculate the average survival rate of all passengers
        self.avg_surv_rate = data['Survived'].mean()
        # Calculate the probability of survivors for each group
        for col in cols:
            # Save the probabilities for later use
            self.surv_lookups[col] = data[['Survived', col]].groupby([col]).mean()

    def preprocess(self, data):
        """Apply learned preprocessing"""
        # Drop the name as it carries no information
        data.drop(['Name'], axis=1, inplace=True)

        # Clean Ticket strings to numbers. Dropped for now.
        data['Ticket'] = data['Ticket'].apply(lambda s: s.split(' ')[-1])
        data.drop(['Ticket'], axis=1, inplace=True)

        # Get the sector value from cabin numbers
        data['Cabin'] = data['Cabin'].apply(lambda s: str(s)[0] if s is not np.nan else None)

        # Replace sex with a boolean value
        data['Sex'].replace({'male': 0, 'female': 1}, inplace=True)

        # Replace the string value with the calculated group probabilities
        cols = ['Cabin', 'Embarked', 'SibSp', 'Parch']
        for col in cols:
            data[col] = data[col].apply(lambda s: self.surv_lookups[col].at[s, 'Survived']
            if s in self.surv_lookups[col].index.values else self.avg_surv_rate)

        # Convert to numpy arrays
        np_data = np.array(data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']])
        np_ids = np.array(data['PassengerId'])

        if 'Survived' in data.columns:
            labels_arr = np.array(data['Survived'])
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

    # Hyperparameter optimisation
    model_prototypes = [{'model': ensemble.HistGradientBoostingClassifier(),
                         'params': {'learning_rate': [0.1],
                                    'max_leaf_nodes': [15],
                                    'min_samples_leaf': [32],
                                    'l2_regularization': [.7],
                                    'max_bins': [30]}}]

    # Initialise data preprocessor and learn parameters
    preprocessor = DataPreprocessor(data=df_all_train, cols=['Cabin', 'Embarked', 'SibSp', 'Parch'])

    # Run an experiment for each split and log the scores
    for m, model_prototype in enumerate(model_prototypes):
        model = GridSearchCV(estimator=model_prototype['model'], param_grid=model_prototype['params'],
                              cv=kf_splits, scoring='f1', refit=True, n_jobs=8)

        train_arr, _, train_labels_arr = preprocessor.preprocess(df_all_train)

        model.fit(train_arr, train_labels_arr)

        model_prototypes[m]['best_score'] = model.best_score_
        model_prototypes[m]['model'] = model

        print(model.best_score_)
        print(model.best_params_)

    # Find best model
    model = sorted(model_prototypes, key=lambda s: s['best_score'])[-1]
    print(model['best_score'])

    # Load and prepare test data
    df_test = pd.read_csv('./titanic/data/test.csv')
    test_arr, test_ids_arr, _ = preprocessor.preprocess(df_test)

    # Make predictions
    pred_arr = model['model'].predict(test_arr)

    # Construct output data format
    out = np.stack([test_ids_arr, pred_arr]).transpose()
    np.savetxt("output.csv", out, comments='', delimiter=",", fmt='%d', header='PassengerId,Survived')

