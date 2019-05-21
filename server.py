from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as numpy

df = pd.read_csv('docs/topics_full_clean.csv', ';')  
df.head()
print(df.count())

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
   
from nltk.stem import PorterStemmer
class PorterTokenizer(object):
    def __init__(self):
        self.wnl = PorterStemmer()
    def __call__(self, articles):
        return [self.wnl.stem(t) for t in word_tokenize(articles)]



topics_text = df['Text']
topics_text = topics_text.str.replace('\d+', '') # for digits
topics_text = topics_text.str.replace(r'(\b\w{1,2}\b)', '') # for words
topics_text = topics_text.str.replace('[^\w\s]', '')

print("Text cleaned")

topics_challenge = df['Challenge']
topics_challenge = topics_challenge.str.replace('\d+', '') # for digits
topics_challenge = topics_challenge.str.replace(r'(\b\w{1,2}\b)', '') # for words
topics_challenge = topics_challenge.str.replace('[^\w\s]', '')

print("Challenge cleaned")

topics_scope = df['Scope']
topics_scope = topics_scope.str.replace('\d+', '') # for digits
topics_scope = topics_scope.str.replace(r'(\b\w{1,2}\b)', '') # for words
topics_scope = topics_scope.str.replace('[^\w\s]', '')

print("Scope cleaned")


topics_impact = df['Impact']
topics_impact = topics_impact.str.replace('\d+', '') # for digits
topics_impact = topics_impact.str.replace(r'(\b\w{1,2}\b)', '') # for words
topics_impact = topics_impact.str.replace('[^\w\s]', '')

print("Impact cleaned")

from sklearn.feature_extraction.text import TfidfVectorizer

##Handle full text search model
vectorizerText = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                 strip_accents = 'unicode', # works 
                                stop_words = 'english', # works
                                lowercase = True, # works
                                max_df = 0.05, # works
                                min_df = 0.001)
trsfm_Text=vectorizerText.fit_transform(topics_text)
print(len(vectorizerText.get_feature_names()))
pd.DataFrame(trsfm_Text.toarray(),columns=vectorizerText.get_feature_names(),index=topics_text)

print("Full text model created")

##Handle challenge part search model
vectorizerChallenge = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                 strip_accents = 'unicode', # works 
                                stop_words = 'english', # works
                                lowercase = True, # works
                                max_df = 0.05, # works
                                min_df = 0.001)
trsfm_Challenge=vectorizerChallenge.fit_transform(topics_challenge)
print(len(vectorizerChallenge.get_feature_names()))
pd.DataFrame(trsfm_Challenge.toarray(),columns=vectorizerChallenge.get_feature_names(),index=topics_challenge)

print("Challenge model created")

##Handle scope part search model
vectorizerScope = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                 strip_accents = 'unicode', # works 
                                stop_words = 'english', # works
                                lowercase = True, # works
                                max_df = 0.05, # works
                                min_df = 0.001)
trsfm_Scope=vectorizerScope.fit_transform(topics_scope)
print(len(vectorizerScope.get_feature_names()))
pd.DataFrame(trsfm_Scope.toarray(),columns=vectorizerScope.get_feature_names(),index=topics_scope)

print("Scope model created")

##Handle impact part search model
vectorizerImpact = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                 strip_accents = 'unicode', # works 
                                stop_words = 'english', # works
                                lowercase = True, # works
                                max_df = 0.05, # works
                                min_df = 0.001)
trsfm_Impact=vectorizerImpact.fit_transform(topics_impact)
print(len(vectorizerImpact.get_feature_names()))
pd.DataFrame(trsfm_Impact.toarray(),columns=vectorizerImpact.get_feature_names(),index=topics_impact)

print("Impact model created")

trsfmT_challenge = vectorizerText.transform(topics_challenge).toarray()

print("Topic's Challenge vectorization done")

trsfmT_scope = vectorizerText.transform(topics_scope).toarray()

print("Topic's Scope vectorization done")

trsfmT_impact = vectorizerText.transform(topics_impact).toarray()

print("Topic's Impact vectorization done")

from sklearn.metrics.pairwise import cosine_similarity


from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    response = "Hello World!"
    return jsonify(response)

@app.route("/", methods=['POST'])
def helloBack():
    if request.is_json:
        content = request.get_json()
        if 'text' in content.keys():
            query=content['text']
            query = query.replace('\d+', '') # for digits
            query = query.replace(r'(\b\w{1,2}\b)', '') # for words
            query = query.replace('[^\w\s]', '')
            if 'type' in content.keys():
                if content['type'] == 'specific_challenge':
                    trsfm_query = vectorizerChallenge.transform([query]).toarray()
                    sims = cosine_similarity(trsfm_query, trsfm_Challenge)
                if content['type'] == 'scopes':
                    trsfm_query = vectorizerScope.transform([query]).toarray()
                    sims = cosine_similarity(trsfm_query, trsfm_Scope)
                if content['type'] == 'expected_impacts':
                    trsfm_query = vectorizerImpact.transform([query]).toarray()
                    sims = cosine_similarity(trsfm_query, trsfm_Impact)
            else:
                trsfm_query = vectorizerText.transform([query]).toarray()
                sims = cosine_similarity(trsfm_query, trsfm_Text)
            top100 = pd.DataFrame(numpy.sort(sims)[0][-100:],df['Title'][numpy.argsort(sims)[0][-100:]])              
            return top100.reset_index().to_json(orient='records')
        else:
            print("json not valid")
            