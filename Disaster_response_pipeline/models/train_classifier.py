import sys
import os
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """
    Loads data from SQL Database
    Args:
    database_filepath: SQL database file
    Returns:
    X: Features dataframe
    Y: Target dataframe
    category_names: Target labels 
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', con = engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenizes a given text.
    Args:
    text: text
    Returns:
    array of clean tokens
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    words = word_tokenize(text)
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds classification model """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(OneVsRestClassifier(RandomForestClassifier())))])
    parameters  = {'vect__ngram_range': ((1, 1), (1, 2)),
                   'tfidf__use_idf': [True, False]}
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=4)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model
    Args:
    model: Trained model
    X_test: Test features
    Y_test: Test labels
    category_names: labels 
    """
    # predict
    Y_preds = model.predict(X_test)
    
    for i in range(len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test.values[:, i], Y_preds[:, i]))

def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle file    
    Args:
    model: Trained model
    model_filepath: Filepath to save the model
    """
    if os.path.exists(model_filepath):
        os.remove(model_filepath)
    pickle.dump(model, open(model_filepath, 'wb'))

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