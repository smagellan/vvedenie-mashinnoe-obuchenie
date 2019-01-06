#! /usr/bin/python3

from sklearn.feature_extraction.text import TfidfVectorizer;
from sklearn.linear_model import Ridge;
from sklearn.feature_extraction import DictVectorizer;
from scipy.sparse import hstack;
import pandas as pd;


def reconstructFeatures(featuresMatrix, tfIdVectorizer, dictVectorizer):
    print("processing 'FullDescription' column with lower");
    featuresMatrix['FullDescription'] = featuresMatrix['FullDescription'].apply(lambda x: x.lower());
    print("processing 'FullDescription' with regex");
    featuresMatrix['FullDescription'] = featuresMatrix['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True);

    print("vectorizing 'FullDescription' with TfidfVectorizer");
    fullDescriptionVectorized = None;
    fullDescrVectorizer = None;
    if tfIdVectorizer == None:
        fullDescrVectorizer = TfidfVectorizer(min_df=5, max_df=1.0);
        fullDescriptionVectorized = fullDescrVectorizer.fit_transform(featuresMatrix['FullDescription'].values);
    else:
        fullDescrVectorizer = tfIdVectorizer;
        fullDescriptionVectorized = tfIdVectorizer.transform(featuresMatrix['FullDescription'].values);

    print("replacing nans for 'LocationNormalized' and 'ContractTime' columns");
    featuresMatrix['LocationNormalized'].fillna('nan', inplace=True);
    featuresMatrix['ContractTime'].fillna('nan', inplace=True);

    print("vectorizing 'LocationNormalized' and 'ContractTime' with DictVectorizer");
    x_categ = None;
    enc = None;
    if dictVectorizer == None:
        enc = DictVectorizer();
        x_categ = enc.fit_transform(featuresMatrix[['LocationNormalized', 'ContractTime']].to_dict('records'));
    else:
        enc = dictVectorizer;
        x_categ = dictVectorizer.transform(featuresMatrix[['LocationNormalized', 'ContractTime']].to_dict('records'));
    print("reconstructing training set with hstack");
    x_vectorized = hstack([fullDescriptionVectorized, x_categ]);
    return [x_vectorized, fullDescrVectorizer, enc];


data_train = pd.read_csv('salary-train.csv');
X_train = data_train[['FullDescription', 'LocationNormalized', 'ContractTime']].copy();
Y_train = data_train['SalaryNormalized'].copy();
[X_train_reconstructed, tfid, enc] = reconstructFeatures(X_train, None, None);


data_test = pd.read_csv('salary-test-mini.csv');
X_test = data_test[['FullDescription', 'LocationNormalized', 'ContractTime']].copy();
[X_test_reconstructed, tfid, enc] = reconstructFeatures(X_test, tfid, enc);

predictor = Ridge(alpha=1, random_state=241);
print("training Ridge");
predictor.fit(X_train_reconstructed, Y_train);
Y_predicted = predictor.predict(X_test_reconstructed);
print("Y predicted: {}".format(Y_predicted));

with open('w4t1a1.txt', 'w') as f:
    f.write("%2.2f %2.2f" % (Y_predicted[0], Y_predicted[1]));