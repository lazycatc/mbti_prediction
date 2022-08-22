import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import re

from nltk.corpus import stopwords
from sklearn.decomposition import PCA
#from spacy.lang.en.stop_words import STOP_WORDS

from bokeh.io import output_notebook, show
from bokeh.models import BoxZoomTool, ColumnDataSource, HoverTool, ResetTool
from bokeh.models.annotations import Span, Label
from bokeh.plotting import figure

from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model, Model

from attention import Attention_layer

#import spacy
#from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Input,GRU, Dense, Dropout,Conv1D, Conv2D, LSTM,Reshape
from tensorflow.keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score


from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from wordcloud import WordCloud, ImageColorGenerator
import pickle
from bokeh.models.widgets import Div
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# load files
stop_words = set(stopwords.words())
w2v = Word2Vec.load('tweet_w2v.model')
model1 = pickle.load(open('model/model1.pkl','rb'))
model2 = pickle.load(open('model/model2.pkl','rb'))
model3 = pickle.load(open('model/model3.pkl','rb'))
model4 = pickle.load(open('model/model4.pkl','rb'))
dic1 = np.load('model/file1.npy', allow_pickle='TRUE').reshape(1)[0]
dic2 = np.load('model/file2.npy', allow_pickle='TRUE').reshape(1)[0]
dic3 = np.load('model/file3.npy', allow_pickle='TRUE').reshape(1)[0]
dic4= np.load('model/file4.npy', allow_pickle='TRUE').reshape(1)[0]
vocab_df = pd.read_pickle("model/vocab_df.pkl")


# data preprocessing
def vectorize_reviews(data, maxlen=1000, embedding_dim=100):
    vectorized_data = np.zeros(shape=(len(data), maxlen, embedding_dim))

    for row, review in enumerate(data):
        # Preprocess each review
        tokens = simple_preprocess(review)

        # Truncate long reviews
        if len(tokens) > maxlen:
            tokens = tokens[:maxlen]

        # Get vector for each token in review
        for col, token in enumerate(tokens):
            try:
                word_vector = w2v.wv[token]
                # Add vector to array
                vectorized_data[row, col] = word_vector[:embedding_dim]
            except KeyError:
                pass

    return vectorized_data

# predict the result
def result(data):
    #data='apple day'
    data = re.sub(r'http[s]?://\S+', ' ', data)
    data = re.sub(r'\(', ' ', data)
    data = re.sub(r'\)', ' ', data)
    data = data.lower().split()
    data = ' '.join(data)
    review=[data]
    x=vectorize_reviews(review)
    y1=  2*model1.predict(x)[0][0]-1
    y2 = 2*model2.predict(x)[0][0]-1
    y3 = 2*model3.predict(x)[0][0]-1
    y4 = 2*model4.predict(x)[0][0]-1
    return [y1,y2,y3,y4]


# plot score pic
def plot_result(result):
    # define data set
    # ESFP
    d1,d2=min(result[0],0),max(result[0],0)
    c1,c2=min(result[1], 0),max(result[1], 0)
    b1,b2=min(result[2], 0), max(result[2], 0)
    a1,a2=min(result[3], 0), max(result[3], 0)
    s1 = pd.Series(['Judging',a1, 'Perceiving', a2])
    s2 = pd.Series(['Thinking', b1, 'Feeling', b2])
    s3 = pd.Series(['Sensing', c1, 'Intuition', c2])
    s4 = pd.Series(['Extroversion', d1, 'Introversion', d2])
    df = pd.DataFrame([list(s1), list(s2), list(s3), list(s4)], columns=['label1', 'value1', 'label2', 'value2'])

    # create subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                        shared_yaxes=True, horizontal_spacing=0)

    fig.append_trace(
        go.Bar(y=df.index, x=df.value1, orientation='h', width=0.4, showlegend=False, marker_color='#4472c4'), 1, 1)
    fig.append_trace(
        go.Bar(y=df.index, x=df.value2, orientation='h', width=0.4, showlegend=False, marker_color='#ed7d31'), 1, 2)
    fig.update_yaxes(showticklabels=False)  # hide all yticks
    annotations = []
    for i, row in df.iterrows():
        if row.label1 != '':
            annotations.append({
                'xref': 'x1',
                'yref': 'y1',
                'y': i,
                'x': row.value1,
                'text': row.value1,
                'xanchor': 'right',
                'showarrow': False})
            annotations.append({
                'xref': 'x1',
                'yref': 'y1',
                'y': i - 0.3,
                'x': -1,
                'text': row.label1,
                'xanchor': 'right',
                'showarrow': False})
        if row.label2 != '':
            annotations.append({
                'xref': 'x2',
                'yref': 'y2',
                'y': i,
                'x': row.value2,
                'text': row.value2,
                'xanchor': 'left',
                'showarrow': False})
            annotations.append({
                'xref': 'x2',
                'yref': 'y2',
                'y': i - 0.3,
                'x': 1,
                'text': row.label2,
                'xanchor': 'left',
                'showarrow': False})

    fig.update_layout(annotations=annotations)
    return fig


# Helper function
def get_coords(word,vocab_df):
    """Given a word from `vocab_df`, returns tuple with x, y coordinates."""
    coords = vocab_df[vocab_df['word'] == word][['x', 'y']]
    return list(coords.itertuples(name=None, index=False))[0]


# Create interactive plot
def interactive_plot(vocab_df,text):
    polar_words1 = [key[0][6:] for key in
                    sorted(dic1.items(), key=lambda item: -item[1])][:25]+[key[0][6:] for key in
                    sorted(dic1.items(), key=lambda item: item[1])][:25]
    polar_words2 = [key[0][6:] for key in
                    sorted(dic2.items(), key=lambda item: -item[1])][:25]+[key[0][6:] for key in
                    sorted(dic2.items(), key=lambda item: item[1])][:25]

    polar_words3 = [key[0][6:] for key in
                    sorted(dic3.items(), key=lambda item: -item[1])][:25]+[key[0][6:] for key in
                    sorted(dic3.items(), key=lambda item: item[1])][:25]

    polar_words4 = [key[0][6:] for key in
                    sorted(dic4.items(), key=lambda item: -item[1])][:25]+[key[0][6:] for key in
                    sorted(dic4.items(), key=lambda item: item[1])][:25]

    text=text.split()
    polar_words1 = [i for i in polar_words1 if i in text]
    polar_words2 = [i for i in polar_words2 if i in text]
    polar_words3 = [i for i in polar_words3 if i in text]
    polar_words4 = [i for i in polar_words4 if i in text]

    v1 = vocab_df[vocab_df['word'].isin(polar_words1)]
    v2 = vocab_df[vocab_df['word'].isin(polar_words2)]
    v3 = vocab_df[vocab_df['word'].isin(polar_words3)]
    v4 = vocab_df[vocab_df['word'].isin(polar_words4)]

    p = figure(plot_width=600,
           plot_height=400,
           tools=[HoverTool(tooltips='@word'), BoxZoomTool(), ResetTool()],
           title='Text Vocabulary')

           # Add vocabulary
    source1 = ColumnDataSource(v1)
    source2 = ColumnDataSource(v2)
    source3 = ColumnDataSource(v3)
    source4 = ColumnDataSource(v4)

    p.circle('x', 'y', source=source1, size=5,
                    fill_color='blue', fill_alpha=0.3,
                    hover_fill_color='yellow')
    p.circle('x', 'y', source=source2, size=5,
             fill_color='yellow', fill_alpha=0.3,
             hover_fill_color='yellow')
    p.circle('x', 'y', source=source3, size=5,
             fill_color='red', fill_alpha=0.3,
             hover_fill_color='yellow')
    p.circle('x', 'y', source=source4, size=5,
             fill_color='green', fill_alpha=0.3,
             hover_fill_color='yellow')

           # Add verticle and horizontal lines
    line_x, line_y = get_coords('like',vocab_df)

    vline = Span(location=line_x, dimension='height',
                    line_dash='dashed', line_color='red')
    p.add_layout(vline)

    hline = Span(location=line_y, dimension='width',
             line_dash='dashed', line_color='red')
    p.add_layout(hline)

           # Display plot
    return p



backgroud_Image = plt.imread('pic/brain.jpg')
def genearte_wc(text):
    wordcloud = WordCloud(background_color='white',
                          width=1000, mask=backgroud_Image, max_words=100,
                          height=860, margin=2).generate(text)
    img_colors = ImageColorGenerator(backgroud_Image)
    # 字体颜色为背景图片的颜色
    wordcloud.recolor(color_func=img_colors)

    plt.imshow(wordcloud)
    plt.title('WordCloud')
    plt.axis("off")
    plt.show()

    #filename = 'wc_{}.jpg'.format(typ)
    #wordcloud.to_file(filename)



