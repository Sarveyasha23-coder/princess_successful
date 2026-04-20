import pandas as pd

def load_data(anime_path, rating_path):
    anime = pd.read_csv(anime_path)
    rating = pd.read_csv(rating_path)

    rating = rating[rating['rating'] != -1]

    df = pd.merge(rating, anime, on='anime_id')

    return df, anime