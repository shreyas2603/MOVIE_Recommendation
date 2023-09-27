from flask import Flask, render_template, request, send_from_directory
import numpy as np
import joblib

app = Flask(__name__, static_url_path='/static')

# Load your data and models
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
cosine_similarities = joblib.load("cosine_similarities.pkl")
index = joblib.load("index_mapping.pkl")
title = joblib.load("title.pkl")

class Movie:
    def __init__(self, Movie_Name):
        self.Movie_Name = Movie_Name

@app.route('/recommend', methods=['POST'])
def recommend_movie():
    movie_name = request.form.get('movie-search')  # Get the movie name from the form
    num_recommendations = 5

    movie_index = index.get(movie_name)
    if movie_index is None:
        return render_template('index.html', prediction=["Sorry, the movie you provided is not present in our current dataset."])

    sim_score = cosine_similarities[movie_index]
    sim_movie_indx = np.argsort(sim_score)[::-1][1:num_recommendations + 1]
    recommendations = [title[i] for i in sim_movie_indx]

    return render_template('index.html', prediction=recommendations)

@app.route('/')
def normal():
    return render_template('index.html', prediction=None)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
