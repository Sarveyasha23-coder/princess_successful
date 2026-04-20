from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_model(anime):
    anime['genre'] = anime['genre'].fillna('')
    anime['type'] = anime['type'].fillna('')

    anime['content'] = anime['genre'] + " " + anime['type']

    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(anime['content'])

    similarity = cosine_similarity(matrix)

    return similarity