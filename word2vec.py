import nltk
from nltk.tokenize import sent_tokenize
from pyvi import ViTokenizer, ViPosTagger
from gensim import corpora, models, similarities
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from text_classifier.tokenizer import *
import matplotlib.pyplot as plt
import math
import pickle as pk
from text_classifier.data_cleaner import DataCleaner

# define constant
number_of_words = 0
interval = 1500
corpus_file = './data/corpus.txt'
version = 10
tf_file = './term_frequency/tf_ver' + str(version) + '.pkl'
dict_file = './dictionary/dict_ver' + str(version) + '.dict'
w2v_file = './word_vector/word2vec_ver' + str(version) + '.model'
with open(corpus_file, encoding="utf-8") as f:
    corpus = f.read().lower()
sentences = sent_tokenize(corpus)
number_of_sentences = len(sentences)

tokenized_corpus = []
print ('start tokenize word using FinTokenizer')

number_of_interval = math.ceil (number_of_sentences/interval)
print (number_of_interval)
for i in range(number_of_interval):
    texts = sentences[i * interval: (i+1)*interval]
    words = tokenize_corpus(texts)
    for i in range(len(words)):
        tokenized_corpus.append(words[i])
tokenized_corpus = np.asarray(tokenized_corpus)

DataCleaner = DataCleaner(tokenized_corpus)
corpus = DataCleaner.clean()
corpus = np.asarray(corpus)


# create dictionary with key is word and value is term frequency
print ('-------------create evaluate tf----------------')
my_dictionary = []
number_times = []
tf_arr = []
t = 0
for sentence in corpus:
    for word in sentence:
        t+=1
        if (word in my_dictionary):
            number_times[my_dictionary.index(word)]+=1
        else:
            my_dictionary.append(word)
            number_times.append(1)

for num in number_times:
    tf_arr.append(num/t)
tf_arr = np.asarray(tf_arr)
min_tf = np.min(tf_arr)
dicts = dict((key, value) for (key, value) in zip(my_dictionary, tf_arr))
with open(tf_file,'wb') as output:
    pk.dump(dicts,output,pk.HIGHEST_PROTOCOL)
print ('-------------evaluate tf done----------------')
# using fastext to create dictionary and word representation 
print ('-------------start using fastect to learn vector representation of word-------------------')


dictionary = corpora.Dictionary(corpus)

dictionary.save(dict_file) # store the dictionary, for future
model = FastText(corpus, size = 50, min_count = 1, window = 2, sg = 0, hs = 1, iter = 10)
model.save(w2v_file)
model = FastText.load(w2v_file)
existent_words = ["đi","cổ phiếu", "mua", "xem","nhỉ","nào","ssi","tên"]
# print ('we have: ', len(dictionary) ,' words in dictionary')
print ('test vector with word : đầu tiên---')

# vec = model.wv[existent_word]
for existent_word in existent_words:    
    similar_words = model.wv.most_similar(existent_word)
    print (existent_word,similar_words)