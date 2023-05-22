from flask import Flask, request, make_response, render_template, url_for
import pickle
import pandas as pd
import numpy as np

df = pd.read_csv('./ml model/imdb_movies.csv')
movies = df["names"].tolist()
ratings = df["score"].tolist()
description = df["overview"].tolist()
movies = list(map(lambda x : x.lower() , movies))
model = pickle.load(open('./ml model/k_mean_model.pickle', 'rb'))
# model = pickle.load(open('./ml model/aglomerative_cluster_model.pickle', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    reply = []
    if request.method == "POST":
        movieName = request.form['movieName']
        index_of_movie = movies.index(movieName.lower())
        c = model.labels_[index_of_movie]
        movies_similar = np.argwhere((model.labels_ == c))
        ans = []
        for i in movies_similar:
            ans.append([movies[i[0]] , ratings[i[0]] , description[i[0]]])
        reply = ans
    
    return render_template('index.html', reply=reply)


if __name__ == '__main__':
    app.run(debug=True)