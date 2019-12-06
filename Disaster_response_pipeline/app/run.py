import json
import plotly
import pandas as pd
import nltk
nltk.download(['stopwords'])

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Layout, Figure
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes a given text.
    Args:
    text: text
    Returns:
    array of clean tokens
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    words = word_tokenize(text)
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def message_by_cat(df):
    """Create a plotly figure of messages per category barplot
    Args:
    df: the dataset
    Returns:
    figure of messages per category barplot
    """

    # Grop by categories
    categories = df.iloc[:, 4:].sum().sort_values(ascending=False)

    data = [Bar(
        x=categories.index,
        y=categories,
        opacity=0.8
    )]

    layout = Layout(
        title="Messages per Category",
        xaxis=dict(
            title='Categories',
            tickangle=45
        ),
        yaxis=dict(
            title='number of Messages',
        )
    )

    return Figure(data=data, layout=layout), categories.index[:10]


def categories_per_genre(df, top_cat):
    """Create a plotly figure of categories per genre stacked barplot
    Args:
    df: the dataset
    top_cat: top categories
    Returns:
    figure categories per genre
    """

    # Grop by categories
    genres = df.groupby('genre').sum()[top_cat]

    color_bar = 'DarkGreen'

    data = []
    for cat in genres.columns[1:]:
        data.append(Bar(
                    x=genres.index,
                    y=genres[cat],
                    name=cat)
                    )

    layout = Layout(
        title="Categories per genre (Top 10)",
        xaxis=dict(
            title='Genres',
            tickangle=45
        ),
        yaxis=dict(
            title='number of messages per Category',
        )
    )

    return Figure(data=data, layout=layout)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # encode plotly graphs in JSON
    graphs = [fig1, fig2]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# get figures and top categories
fig1, top_cat = message_by_cat(df)
fig2 = categories_per_genre(df, top_cat)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()