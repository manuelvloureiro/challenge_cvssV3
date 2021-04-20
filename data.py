from pathlib import Path
import pandas as pd
import requests
import zipfile
from tqdm import tqdm
import json
from itertools import chain
from vectorizer import Vectorizer
from sklearn.model_selection import train_test_split

DATA = Path('data')
DATA.mkdir(parents=True, exist_ok=True)

URL = 'https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{}.json.zip'


def download_url(url, filename):
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)


def unzip(filename, directory):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(directory)


def get_data(start=2002, end=2022):
    iterator = tqdm(
        range(start, end),
        desc='Downloading and unzipping data',
        ascii=True
    )
    for year in iterator:
        url = URL.format(str(year))
        filename = DATA / url.split('/')[-1]
        if not filename.exists():
            download_url(url, filename)
        unzip(filename, DATA)
        filename.unlink()


def flatten(o):
    return list(chain.from_iterable(o))


def readjson(path):
    with open(path) as f:
        obj = f.read()
    return json.loads(obj)


def process_impact(impact):
    output = {}

    cvssv3 = impact.get('baseMetricV3', {}).get('cvssV3', {})
    cvssv3 = {'cvssV3_' + k: v for k, v in cvssv3.items()}
    output.update(cvssv3)

    cvssv2 = impact.get('baseMetricV2', {}).get('cvssV2', {})
    other_cvssv2 = {'cvssV2_' + k: v for k, v in
                    impact.get('baseMetricV2', {}).items() if k != 'cvssV2'}
    cvssv2 = {'cvssV2_' + k: v for k, v in cvssv2.items()}
    output.update(cvssv2)
    output.update(other_cvssv2)

    return output


def prep_text(text):
    text = text.lower()
    return text


def load_data(test_size=0.3, seed=42, vectorize_descriptions=True, **kwargs):
    if len(list(DATA.glob('*'))) == 0:
        get_data()

    data = []

    files = tqdm(
        list(DATA.glob('*')),
        desc='Loading data',
        ascii=True
    )


    for file in files:
        data.extend(readjson(file).get('CVE_Items', []))

    impact = [process_impact(o.get('impact', {})) for o in data]
    columns = list(set(flatten(list(o.keys()) for o in impact)))
    df = pd.DataFrame(impact, columns=columns)

    if vectorize_descriptions:
        df['description'] = \
            [prep_text(o['cve']['description']['description_data'][0]['value'])
             for o in data]

    print('Number of CVEs:', df.shape[0])

    df = df.dropna(subset=['cvssV2_version'])
    print('Number of non-rejected CVEs:', df.shape[0])

    predictions_boolean = df['cvssV3_version'].isna()
    missing = df[predictions_boolean]
    print('Number of CVEs missing CVSSv3:', missing.shape[0])

    df = df[~predictions_boolean]

    # remove features that are not necessary
    del df['cvssV2_vectorString']
    del df['cvssV2_version']
    del df['cvssV3_vectorString']
    del df['cvssV3_version']

    print('Number of CVEs with CVSSv3:', df.shape[0])

    train, test = train_test_split(
        df,
        test_size=int(test_size * df.shape[0]),
        random_state=seed
    )
    test, valid = train_test_split(
        test,
        test_size=int(.5 * test.shape[0]),
        random_state=seed
    )

    print('Number of CVEs for training:', train.shape[0])
    print('Number of CVEs for testing:', test.shape[0])
    print('Number of CVEs for validation:', valid.shape[0])

    def extend_df_with_vectorizer(df_, column='description'):
        desc = v.to_df(df_[column])
        df_ = df_.drop(axis=1, columns=['description'])
        return pd.concat([df_.reset_index(drop=True), desc], axis=1)

    if vectorize_descriptions:
        v = Vectorizer(**kwargs)
        v.fit(train['description'])

        train = extend_df_with_vectorizer(train)
        test = extend_df_with_vectorizer(test)
        valid = extend_df_with_vectorizer(valid)
        missing = extend_df_with_vectorizer(missing)

    cvssv3_columns = [o for o in train.columns if 'cvssV3' in o]
    predict_columns = list(set(train.columns).difference(cvssv3_columns))

    return {
        'predict': {
            'train': train[predict_columns],
            'test': test[predict_columns],
            'valid': valid[predict_columns],
            'missing': missing[predict_columns]
        },
        'cvssV3': {
            'train': train[cvssv3_columns],
            'test': test[cvssv3_columns],
            'valid': valid[cvssv3_columns]
        }
    }
