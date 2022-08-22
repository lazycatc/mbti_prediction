import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import plotly.figure_factory as ff
from wordcloud import WordCloud, ImageColorGenerator
import base64
import pickle
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import re

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
import model
from model import result, plot_result
from bokeh.models.widgets import Div
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('pic/blue.jpg')
st.title('MBTI Prediction Using Text Data')
#st.header("# Streamlit示例")

#
#So for example, someone who prefers introversion, intuition, thinking and perceiving would be labelled an INTP in the MBTI system, and there are lots of personality based components that would model or describe this person’s preferences or behaviour based on the label.
#It is used in businesses, online, for fun, for research and lots more. "
#link='[Explore MBTI](https://retailscope.africa/)'
#st.markdown("Perception involves all the ways of becoming aware of things, people, happenings, or ideas. Judgment involves all the ways of coming to conclusions about what has been perceived. If people differ systematically in what they perceive and in how they reach conclusions, then it is only reasonable for them to differ correspondingly in their interests, reactions, values, motivations, and skills.")
st.markdown("The Myers Briggs Type Indicator (or MBTI for short) is one of the most popular personality test in the world. " 
            "It is a personality type system that divides everyone into 16 distinct personality"
            "types across 4 axis: Introversion (I)-Extroversion (E), Intuition (N)-Sensing (S), "
            "Thinking (T)-Feeling (F), Judging (J)-Perceiving (P). Explore MBTI for more information using the following link.")
# button
if st.button('Explore MBTI'):
    js = "window.open('https://www.16personalities.com/personality-types')"  # New tab or window
    js = "window.location.href = 'https://www.16personalities.com/personality-types'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

st.markdown("There are many questionnaires to test your MBTI personality type online, but here we can "
            "test your or your friend' or anyone's MBTI personality type just using the text data that is delivered "
            "by this person, e.g. like tweet posts.")
st.subheader("Enter the text you'd like to analyze.")
#st.button('Start the test!')
text= st.text_input('Test result would be moure accuracy if enter more posts. Approximately 30-50 posts.')
if text:
    r=result(text)
    #st.write(r)
    for i in range(4):
        r[i]=round(r[i],2)
    l=[0,0,0,0]
    l[0]='E' if r[0]>0 else 'I'
    l[1]='S' if r[1] > 0 else 'N'
    l[2]='F' if r[2] > 0 else 'T'
    l[3]='P' if r[3] > 0 else 'J'
    a='Your test result is '+''.join(l)+'.'
    st.write(a)

    # 1. display result score
    fig = plot_result(r)
    st.plotly_chart(fig)

    # 2. display result pic
    filename='pic/'+''.join(l)+'.jpg'
    image = Image.open(filename)
    st.image(image, caption='Sunrise by the mountains', use_column_width=True)


    # 3. display interactive plot
    #st.write('Siginificant words')
    #vocab_df= pd.read_pickle("model/vocab_df.pkl")
    #p=model.interactive_plot(vocab_df, text)
    #st.bokeh_chart(p, use_container_width=True)




# wordclound image
# backgroud_Image = plt.imread('pic/brain.jpg')
# wordcloud = WordCloud(background_color='white',
# width=1000, mask=backgroud_Image, max_words=100,
# height=860, margin=2).generate(text)
# img_colors = ImageColorGenerator(backgroud_Image)
# 字体颜色为背景图片的颜色
# wordcloud.recolor(color_func=img_colors)'''

# Display the generated image:
# plt.figure(figsize=(100, 100))
# fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]})
# axes[0].imshow(wordcloud, interpolation='bilinear')
# axes[1].imshow(backgroud_Image, cmap=plt.cm.gray, interpolation="bilinear")
# for ax in axes:
# ax.set_axis_off()
# plt.show()
# st.pyplot(fig)