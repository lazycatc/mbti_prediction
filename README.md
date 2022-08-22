Using Text Data to Predict MBTI Personality


1. Summary: This project is intended to use NLP and neutral network models to predict people’s MBTI personality type using the text data produced by them. 

2. Problem Statement: MBTI, short for Myers-Briggs Type Indicator, is a personality type system that divides everyone into 16 distinct personality types across 4 axis. Today it is a sophisticated and common way to quickly distinguish how people are similar and different.
Nowadays the most common way to find out our personality type is to visit personality type websites and fill out a questionnaire. Answering so many questions is already a big time, not to mention that usually we  cannot in some sense to clearly identify by how much we agree/disagree with the question. 
This project aims to help people to quickly find out MBTI personality just using the text data, e.g. the twitter posts, the messages. There is no need to answer any question, just need to input the text data into our webapp, then your MBTI type result will automatically be there! 

3. Project: The dataset for training is sourced from Kaggle, making use of forum posts from personalitycae.com, consisting of over 8600 tweet users’ recent 50 tweet posts and their MBTI type.
We firs did a data exploration to see there is no missing values, all the data are text, and the labels are imbalanced. Next we did data preprocessing, specifically we did: 1.	Removing the weblinks, MBTI reference (e.g. INTJ), mentions (e.g. @someone), hashtags (e.g. #helloworld), 2.	splitting the target data from 16 categories into 4 binary classifiers, 3.	lemmatization,   4.	word embedding to transform the text data into numeric features,  5.	In order for all the feature array to have the same length, we truncated the feature array to an array of length 1000

4. The final model used is a RNN model, a combination of a LSTM layer and an attention layer, so I call this model LSTM-Attention model. The model information is listed in the following picture. And this model gets more than 85% validation accuracy in each of the binary classification for 4 axes of MBTI.

5. Finally we use the streamlit tool (main code in app.py) to build an interactive website, where you can input the text data then you will see the MBTI prediction results.
 
6. Deliverables: Github: https://github.com/lazycatc/mbti_prediction

