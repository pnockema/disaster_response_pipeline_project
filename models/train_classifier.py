import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans

import pickle

def load_data(database_filepath):
    '''
    Load data from database.

    Input: 
    database_filepath: sql database filepath

    Output:
    X: Feature variable for ML model
    Y: Target variable for ML model
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.DataFrame(engine.connect().execute(text('SELECT * FROM disaster_messages')))
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns


def tokenize(text):
    '''
    Tokenize text as NLP preparation.

    Input:
    text: Text string of any length

    Output:
    clean_tokens: lemmatized and case normalized text tokens
    '''
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Build machine learning model

    Input: None

    Output: GridsearchCV-function of Machine Learning pipeline.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    param_grid = {
    'clf__estimator__n_estimators': [100, 200, 300],
    'clf__estimator__max_depth': [None, 5, 10],
    'clf__estimator__min_samples_split': [2, 5, 10]
    }

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    return grid_search 


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Print evaluation of ML model

    Input:
    model: ML model from build_model but after fitting
    X_test: test input data from train-test-split
    Y_test: test output data from train-test-split
    category_names: Category names to show classification report for

    Output:
    None
    '''
    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    for col in Y_pred.columns:
        print(col)
        print()
        print(classification_report(Y_test[col], Y_pred[col], zero_division=0.0))
    pass


def save_model(model, model_filepath):
    '''
    Save ML model to pickle file

    Input:
    model: trained model from build_model
    model_filepath

    Output:
    None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()