# import neccessary packages
from nltk.stem import WordNetLemmatizer
import io
import random
import string
import nltk
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
nltk.download('popular', quiet=True)

# uncomment the following only the first time
# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only

# will use the wikipedia page on chatbots as our training data to use
f = open('chatbotWiki.txt', 'r', errors='ignore')
# read file and convert it to lower case
raw = f.read().lower()

# convert text to tokens
# list of sentences
sentence_tokens = nltk.sent_tokenize(raw)
# list of words
word_tokens = nltk.word_tokenize(raw)

# will lemmatize tokens and return normalized tokens
lemmer = nltk.stem.WordNetLemmatizer()
# Wordnet is a semantically-oriented dictionary of English included in NLTK


def LemmantizeTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemmantizeTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# handle greeting of the bot
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up",
                   "hey", "yello", "heyo", "what's good", "how's it going")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there",
                      "hello", "I am glad! You are talking to me"]

# will read response and check if greeting is found in the current GREETING_INPUTS we have, and return a random GREETING_RESPONSE


def greeting(greet):
    for word in greet.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
