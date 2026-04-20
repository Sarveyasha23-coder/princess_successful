import pandas as pd

def recommend(anime, similarity, title):
    indices = pd.Series(anime.index, index=anime['name']).drop_duplicates()

    if title not in indices:
        return "Anime not found"

    idx = indices[title]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]

    anime_indices = [i[0] for i in scores]

    return anime['name'].iloc[anime_indices]