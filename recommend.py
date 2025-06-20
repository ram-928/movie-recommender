import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")
df['overview'] = df['overview'].fillna('')

# TF-IDF Matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Reverse mapping
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(title, num=5):
    if title not in indices:
        return f"Movie '{title}' not found."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Example run
if __name__ == "__main__":
    print("Recommendations for 'The Dark Knight':")
    print(recommend("The Dark Knight"))
