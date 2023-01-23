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


"""
method of Generating response: 
Bag of Words
-after we preprocess the text into an array/list of numbers, the bag of words will allow us to keep track of the occurence of 
certain words within the document
-we draw a conclusion of the meaning of words simply from the occurence and content of the words and not by the ordering

will use in combination with the TF-IDF Approach (Term Frequency-Inverse Document Frequency)
-one of the problems with the bag of words approach is that high frequency of certain words may start to dominate the document and we may lose out on 
informational content, words such as "the", "a", "and"
-we can combat this by rescaling the frequency of words by how often they appear in all documents so that the scores for frequent words like "the" 
that are super frequent are penalized

#scores the frequency of a word in a document
TF = (number of times the term appears in the document)/ (Number of  terms in the document)

#scores how rare the word is accross documents
IDF = 1+log(N/n), where N is number of documents, n is number of documents the term t has appeared in

We then use Cosign similarity to determine how important the word is to a document
cosine similarity (d1, d2) = Dot product(d1, d2)/||d1|| * ||d2||
d1, and d2 are non 0 vectors

We will use the user_response we get and look for keywords in the documenent and return one of the several possible responses, if we cant find any 
matching keywords matching the user_response, we return "Sorry, That is outside of my knowledge, please ask another Question or rephrase it"
"""


def response(user_response):
    eliza_response = ''
    sentence_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(
        tokenizer=LemmantizeTokens, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        eliza_response = eliza_response + \
            "Sorry, That is outside of my knowledge, Please rephrase or ask a different question!"
        return eliza_response
    else:
        eliza_response = eliza_response+sentence_tokens[idx]
        return eliza_response


# handle what the bot says initially and tell the user how they can exit the chatbot
flag = True
print("ELIZA: My name is ELIZA. I will answer your queries about Chatbots. When you are finished, type Bye!")
while (flag == True):
    user_response = input()
    user_response = user_response.lower()
    if (user_response != 'bye'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("ELIZA: You are welcome...")
        else:
            if (greeting(user_response) != None):
                print("ELIZA: " + greeting(user_response))
            else:
                print("ELIZA: ", end="")
                print(response(user_response))
                sentence_tokens.remove(user_response)
    else:
        flag = False
        print("ELIZA: I enjoyed our chat, Bye now!")
