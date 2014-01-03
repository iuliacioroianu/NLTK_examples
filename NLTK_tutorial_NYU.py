
"""
Created on Mon Apr 22 15:57:30 2013
File name: NLTK_presentation_code.py 
Author: Iulia Cioroianu
Based on: "Natural Language Processing with Python" Bird, Klein and Loper
Purpose: Intro to text analysis in Python - NLTK; NYU workshop
Data Used: NLTK data and book examples, own UK Manifesto data
"""

#1. Getting started
#Import data for examples

import nltk
nltk.download()
#In the "NLTK Downloader" select the packages and corpora you want to download. 



########################
# Review basic Python: #
########################

#Strings
monty = "Monty Python's \
         Flying Circus."
monty*2 + "plus just last word:" + monty[-7:]
monty.find('Python') #finds position of substring within string
monty.upper() +' and '+ monty.lower()
monty.replace('y', 'x')

# Lists
# As opposed to strings, lists are flexible about the elements they contain. 
sent1 = ['Monty', 'Python']
sent2 = ['and', 'the', 'Holy', 'Grail']
len(sent2)
sent1[1]
sent2.append("1975")
sent1 + sent2
sorted(sent1 + sent2)
' '.join(['Monty', 'Python'])
'Monty Python'.split()

# Import text4 of book examples, Inaugural Addresses
from nltk.book import text4

# Operating on every element. List comprehension.  
len(set([word.lower() for word in text4 if len(word)>5]))
[w.upper() for w in text4[0:5]]

for word in text4[0:5]:
    if len(word)<5 and word.endswith('e'):
        print word, ' is short and ends with e'
    elif word.istitle():
        print word, ' is a titlecase word'
    else:
        print word, 'is just another word'

#Searching text
text4.concordance("vote")
#What other words appear in a similar range of contexts? 
text4.similar("vote")
#  examine just the contexts that are shared by two or more words
text4.common_contexts(["war", "freedom"])

# Counting Vocabulary: the length of a text from start to finish, 
# in terms of the words and punctuation symbols that appear. All tokens. 
len(text4)
#  How many distinct words does the book of Genesis contain? 
# The vocabulary of a text is just the set of tokens that it uses. 
len(set(text4)) #types
len(text4) / len(set(text4)) # Each word used on average x times. Richness of the text. 

#Location of a word in the text: how many spaces from the beginning does it appear? 
#This positional information can be displayed using a dispersion plot. 
#You need NumPy and Matplotlib. 
text4.dispersion_plot(["citizens", "democracy", "freedom", "war", "America", "vote"])

# Generating some random text in the various styles we have just seen.
text4.generate()

# count how often a word occurs in a text,
text4.count("democracy")
# compute what percentage of the text is taken up by a specific word
100 * text4.count('democracy') / len(text4)

# Define functions: 
def lexical_diversity(text):
    return len(text) / len(set(text))

def percentage(count, total):
    return 100 * count / total

lexical_diversity(text4)
percentage(text4.count('a'), len(text4))

# Simple statistics
from nltk import FreqDist
# Counting Words Appearing in a Text (a frequency distribution)
fdist1 = FreqDist(text4)
fdist1
vocabulary1 = fdist1.keys() # list of all the distinct types in the text
vocabulary1[:3] # look at first 3

#words that occur only once, called hapaxes 
fdist1.hapaxes()[:20]

# Words that meet a condition, are long for example
V = set(text4)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

#finding words that characterize a text, relatively long, and occur frequently
fdist = FreqDist(text4)
sorted([w for w in set(text4) if len(w) > 7 and fdist[w] > 7])

# Collocations and Bigrams. 
# A collocation is a sequence of words that occur together unusually often. 
# Built in collocations function
text4.collocations()


#############
#Corpus data#
#############

# Inaugural Address Corpus

from nltk.corpus import inaugural
inaugural.fileids()[:2]
[fileid[:4] for fileid in inaugural.fileids()]

#How the words America and citizen are used over time.

cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'war']
    if w.lower().startswith(target))
cfd.plot()
#cfd.tabulate()

from nltk.corpus import brown
news_words=brown.words(categories="news")
print(news_words)
freq= nltk.FreqDist(news_words)
freq.plot(30)

from nltk import FreqDist
verbs=["should", "may", "can"]
genres=["news", "government", "romance"]
for g in genres:
    words=brown.words(categories=g)
    freq=FreqDist([w.lower() for w in words if w.lower() in verbs])
    print g, freq


# Import own corpus

from nltk.corpus import PlaintextCorpusReader
corpus_root = "C:/Data/Files/"
wordlists = PlaintextCorpusReader(corpus_root, '.*\.txt')
wordlists.fileids()[:3]
wordlists.words('UK_natl_2010_en_Lab.txt')

# Stopwords
	
from nltk.corpus import stopwords
stopwords.words('english')

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)

content_fraction(nltk.corpus.reuters.words())
content_fraction(nltk.corpus.inaugural.words())

# Translator
from nltk.corpus import swadesh
languages = ['en', 'ro', 'es', 'fr', 'pt', 'la']
for i in [100, 141, 143]:
    print swadesh.entries(languages)[i]

# Wordnet 
#dictionary of English

from nltk.corpus import wordnet as wn
wn.synsets('motorcar')
wn.synset('car.n.01').lemma_names
wn.synset('car.n.01').definition
for synset in wn.synsets('car')[1:3]:
    print synset.lemma_names

# Depth of a synset
wn.synset('whale.n.02').min_depth()
wn.synset('vertebrate.n.01').min_depth()
wn.synset('walk.v.01').entailments() #Walking involves stepping

# Hyponims and hypernims
motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
types_of_motorcar[0:2]

##################################
### Importing and accessing text#
#################################

import nltk, re, pprint

# Online books

from urllib import urlopen

url = "http://www.gutenberg.org/files/61/61.txt"
raw = urlopen(url).read()
type(raw)
len(raw)
raw[:75]

tokens = nltk.word_tokenize(raw)
type(tokens)
len(tokens)
tokens[80:110]

text = nltk.Text(tokens)
text.collocations()

# Online articles
    
url = "http://www.bbc.co.uk/news/science-environment-21471908"
#Getting text out of HTML is a sufficiently common task that NLTK provides a helper function nltk.clean_html(), which takes an HTML string and returns raw text.
html = urlopen(url).read()
html[:60]
raw = nltk.clean_html(html)
tokens = nltk.word_tokenize(raw)
tokens[:15]

#Processing RSS Feeds: import feedparser

# Reading local files

f = open('C:\Data\Files\UK_natl_2010_en_Lab.txt')
raw = f.read()
print raw[:100]

# User input
s = raw_input("Enter some text: ")

#Regular exppressions applications. Find and count all vowels.  
import re
word = 'supercalifragilisticexpialidocious'
len(re.findall(r'[aeiou]', word))

# Normalize - ignore upper case
# Tokenize - divide into tokens

f = open('C:\Data\Files\UK_natl_2010_en_Lab.txt')
raw = f.read()
tokens = nltk.word_tokenize(raw)
tokens[:10]
set(w.lower() for w in tokens)

# Stemming - strip off affixes
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
[porter.stem(t) for t in tokens]
[lancaster.stem(t) for t in tokens]

# Lemmatizing - the word is from a dictionary
wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]

# Sentence segmentation
sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
sents = sent_tokenizer.tokenize(raw)
pprint.pprint(sents[182:185])

# Writing output to file
output_file = open('C:\Data\Files\output.txt', 'w')
words = set(sents)
for word in sorted(words):
   output_file.write(word + "\n")
   
##############   
#POS Tagging #
##############

nltk.corpus.brown.tagged_words()

text = nltk.word_tokenize("And now for something completely different")
nltk.pos_tag(text)

text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
nltk.pos_tag(text)

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)

bigram_tagger = nltk.BigramTagger(train_sents)
bigram_tagger.tag(brown_sents[2007])
unseen_sent = brown_sents[4203]
bigram_tagger.tag(unseen_sent)
bigram_tagger.evaluate(test_sents)

###################
# Classification #
##################

######################################
# Names-gender identification example#
######################################

def gender_features(word):
    return {'last_letter': word[-1]}

gender_features('Shrek')

from nltk.corpus import names
import random
names = ([(name, 'male') for name in names.words('male.txt')] +
          [(name, 'female') for name in names.words('female.txt')])
random.shuffle(names)

featuresets = [(gender_features(n), g) for (n,g) in names]
from nltk.classify import apply_features # use apply if you're working with large corpora
train_set = apply_features(gender_features, names[500:])
test_set = apply_features(gender_features, names[:500])

classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))
print nltk.classify.accuracy(classifier, test_set)

classifier.show_most_informative_features(5)

#########################
# Movie reviews example #
#########################

from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.keys()[:2000] # [_document-classify-all-words]

def document_features(document): # [_document-classify-extractor]
    document_words = set(document) # [_document-classify-set]
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

print document_features(movie_reviews.words('pos/cv957_8737.txt'))
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set) 
classifier.show_most_informative_features(5)


