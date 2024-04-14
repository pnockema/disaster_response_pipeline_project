import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load the csv files of messages and categories into Dataframes 
    and merge the Dataframes on 'id' column.

    Input: Paths of CSV-files

    Output: 
    df: Merged Dataframe containing the data of both CSV-files
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    Clean merged df from load_data.

    Input: 
    df: Merged pandas DataFrame from load_data function.

    Output:
    df: Cleaned Dataframe containing one column for each disaster help category. 
    These columns contain 0's and 1's. 1's if text messages is estimated to belong to category.
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand = True)

    #select first row from cetagories
    row = categories.loc[0, :]

    #extract category-names from first row
    category_colnames = row.str.extract('^(.+)-\d$', expand = False)

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.extract('^.+-(\d)$', expand = False)
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns = ['categories'], inplace = True)

    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    engine = create_engine(database_filename)
    df.to_sql('disaster_messages', engine, index=False)
    pass  


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