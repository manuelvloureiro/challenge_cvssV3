import lightgbm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path


class Model:

    def __init__(self, data):
        self.data = data

        self.model = None
        self.y_encoder = LabelEncoder()
        self.multiclass = False

        self.categorical = [
            o for o in data['predict']['train'].columns if
            isinstance(data['predict']['train'].iloc[0][o], str)]

        self.boolean = [o for o in data['predict']['train'].columns if
                        isinstance(data['predict']['train'].iloc[0][o], bool)]

        self.train = self.__prep_dataset(self.data['predict']['train'])
        self.test = self.__prep_dataset(self.data['predict']['test'])

    def __prep_dataset(self, df):
        labelencoder = LabelEncoder()
        for feature in self.categorical:
            df[feature] = labelencoder.fit_transform(df[feature])
            df[feature] = pd.Series(df[feature], dtype="int")

        for feature in self.boolean:
            df[feature] = df[feature].fillna(False).astype(int)

        return df

    def fit(self, target='cvssV3_scope', **kwargs):

        y_train = pd.Series(
            self.y_encoder.fit_transform(self.data['cvssV3']['train'][target]))
        y_test = pd.Series(
            self.y_encoder.transform(self.data['cvssV3']['test'][target]))

        # these hyperparameters are sensible defaults
        parameters = {
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': 'true',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 20,
            'learning_rate': 0.05,
            'verbose': 0,
            'random_state': 42
        }
        parameters.update(kwargs)
        if len(set(y_train)) > 2:
            self.multiclass = True
            parameters['objective'] = 'multiclass'
            parameters['num_class'] = len(set(y_train))
            parameters['metric'] = 'auc_mu'
        else:
            self.multiclass = False

        print(parameters)

        self.model = lightgbm.train(
            parameters,
            lightgbm.Dataset(self.train, label=y_train),
            valid_sets=lightgbm.Dataset(self.test, label=y_test),
            num_boost_round=5000,
            early_stopping_rounds=100
        )

    def predict(self, dataset):
        predictions = self.model.predict(self.__prep_dataset(dataset))
        if self.multiclass:
            labels = [self.y_encoder.classes_[o.argmax()] for o in predictions]
        else:
            labels = [self.y_encoder.classes_[round(o)] for o in predictions]
        return labels, predictions

    def predict_summary(self, dataset, y_true):
        labels, predictions = self.predict(dataset)
        metrics = precision_recall_fscore_support(y_true, labels)
        summary = {
            'accuracy': accuracy_score(y_true, labels),
            'precision': dict(zip(self.y_encoder.classes_, metrics[0])),
            'recall': dict(zip(self.y_encoder.classes_, metrics[1])),
            'fscore': dict(zip(self.y_encoder.classes_, metrics[2])),
            'support': dict(zip(self.y_encoder.classes_, metrics[3]))
        }
        return labels, predictions, summary


if __name__ == '__main__':
    from data import load_data
    from pathlib import Path
    from pprint import pprint

    OUTPUT = Path('output')
    OUTPUT.mkdir(exist_ok=True)

    data = load_data(vectorize_descriptions=True, max_features=1_000)
    model = Model(data)


    def train_predict_summary(target, valid, missing):
        y_true = data['cvssV3']['valid'][target]
        model.fit(target)
        _, _, valid_summary = model.predict_summary(valid, y_true)
        missing_labels_, _ = model.predict(missing)
        return missing_labels_, valid_summary


    def train_predict_summary_all():
        valid = data['predict']['valid'].copy()
        missing = data['predict']['missing'].copy()
        targets = ['scope', 'confidentialityImpact', 'integrityImpact',
                   'availabilityImpact']
        missing_labels_ = {}
        summaries_ = {}
        for target in targets:
            missing_labels_['pred_' + target], summaries_[
                target] = train_predict_summary(
                'cvssV3_' + target, valid, missing)
        return missing_labels_, summaries_


    missing_labels, summaries = train_predict_summary_all()
    results = pd.concat([data['predict']['missing'].filter(regex='cvssV2'),
                         pd.DataFrame(missing_labels)], axis=1)
    results.to_csv(OUTPUT / 'results.csv', index=False)

    pprint(summaries)
