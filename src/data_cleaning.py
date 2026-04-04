import pandas as pd


def load_dataset(path):
    df = pd.read_csv(path)
    return df


def clean_dataset(df):

    df = df.copy()

    df['director'] = df['director'].fillna("Unknown")
    df['cast'] = df['cast'].fillna("Unknown")
    df['country'] = df['country'].fillna("Unknown")

    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month

    df['duration_int'] = df['duration'].str.extract('(\d+)')
    df['duration_int'] = pd.to_numeric(df['duration_int'], errors='coerce')

    return df


def get_genre_counts(df):

    genres = df['listed_in'].str.split(',', expand=True).stack()
    genres = genres.str.strip()

    return genres.value_counts()