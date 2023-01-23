# Simple ChatBot

Building a Simple Chatbot from scratch using Python using NLTK (Natural Language Tool Kit) <br>

# About

Uses the Wikipedia page on chatbots to use as data to pull responses from https://en.wikipedia.org/wiki/Chatbot<br>
Takes in a response from the user and preprocesses it into useful information for the model to understand <br>
uses keyword matching, bag of words, and the TF-IDF appraoch to generate a response for the user

## Pre-requisites

**NLTK(Natural Language Toolkit)**

[Natural Language Processing with Python](http://www.nltk.org/book/) provides a practical introduction to programming for language processing.

For platform-specific instructions, read [here](https://www.nltk.org/install.html)

### Installation of NLTK

```
pip install nltk
pip install scikit-learn
```

### Installing required packages

After NLTK has been downloaded, import/install the required packages

```
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading popular packages
nltk.download('punkt')
nltk.download('wordnet')
```

## How to run

1. installed required packages <br>
2. run via termial

```
python3 chatbot.py
```

or

```
python chatbot.py
```
