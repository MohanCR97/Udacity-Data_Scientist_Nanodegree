import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Takes input messages and categories files, then merge them to a dataframe
    Args:
    messages_file_path: Messages CSV file
    categories_file_path: Categories CSV file
    Returns:
    df: Dataframe obtained from merging the two inputs
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Cleans the dataframe obtained from load_data
    Args:
    df: Dataframe obtained from load_data
    Returns:
    df: Cleaned Dataframe
    """
    categories = df['categories'].str.split(";",expand = True)
    row = categories.iloc[0,:].values
    category_colnames =  [r[:-2] for r in row]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    df.drop(['categories'], axis = 1, inplace = True)
    df[categories.columns] = categories
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    """
    Saves cleaned data from clean_data to an SQL database
    Args:
    df: Dataframe obtained from clean_data
    database_filename: File path of SQL Database to be saved
    """
    if os.path.exists(database_filename):
        os.remove(database_filename)
        
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()