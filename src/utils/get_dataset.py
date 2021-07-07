import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
AVA_DATASET_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'AVA')
AVA_FILE = os.path.join(AVA_DATASET_DIR,'AVA.txt')

columns = [
    "index",
    "image_id",
    "count_rating_1",
    "count_rating_2",
    "count_rating_3",
    "count_rating_4",
    "count_rating_5",
    "count_rating_6",
    "count_rating_7",
    "count_rating_8",
    "count_rating_9",
    "count_rating_10",
    "tag_1",
    "tag_2",
    "challange_id",
]
    
def get_rating_columns():
    return [x for x in columns if x.startswith('count_rating')]


def __get_max_rating(df_row):
    row = df_row[get_rating_columns()]
    max_value_id = row.idxmax()
    max_rating = max_value_id.replace('count_rating_','')
    return max_rating

def get_ava_dataframe(dataset_path=None):
    df = pd.read_csv(AVA_FILE, sep=' ', header=None, names=columns, ) 
    df['rating'] = df.apply(lambda row: __get_max_rating(row), axis=1)
    return df
    
