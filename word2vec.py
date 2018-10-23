import nltk
from nltk.tokenize import sent_tokenize
from pyvi import ViTokenizer, ViPosTagger
from gensim import corpora, models, similarities
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from tokenizer import *
import matplotlib.pyplot as plt
import math
import pickle as pk
# sentences = ["tôi muốn mua 100 cổ phiếu ssi giá 12.4."]
# all_tokens = finTokenizer(sentences[0])
# print (all_tokens)
corpus_file = './data/corpus.txt'
from text_classifier.data_cleaner import DataCleaner
with open(corpus_file, encoding="utf-8") as f:
    corpus = f.read().lower()
    print("----------------------------------CORPUS----")
sentences = sent_tokenize(corpus)
number_of_sentences = len(sentences)
print (len(sentences))
print ('start preprocessing data')
# lol
tokenized_corpus = []
print ('start tokenize word using FinTokenizer')
number_of_words = 0
interval = 1500
number_of_interval = math.ceil (number_of_sentences/interval)
print (number_of_interval)
# lol
for i in range(number_of_interval):
    texts = sentences[i * interval: (i+1)*interval]
    # print (texts)
    words = tokenize_corpus(texts)
    # print (words.shape)
    for i in range(len(words)):
        tokenized_corpus.append(words[i])
tokenized_corpus = np.asarray(tokenized_corpus)
print (tokenized_corpus[0:10])
DataCleaner = DataCleaner(tokenized_corpus)
corpus = DataCleaner.clean()
corpus = np.asarray(corpus)
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

print (my_dictionary)
for num in number_times:
    tf_arr.append(num/t)
tf_arr = np.asarray(tf_arr)
min_tf = np.min(tf_arr)
print (min_tf)
# for i,word in enumerate(my_dictionary):
#     if(DataCleaner.is_stop_word(word)):
#         tf_arr[i] = min_tf
dicts = dict((key, value) for (key, value) in zip(my_dictionary, tf_arr))
# plt.hist(tf_arr, bins='auto') 
# plt.show()
# print (dicts)
with open('tf_dicts.pkl','wb') as output:
    pk.dump(dicts,output,pk.HIGHEST_PROTOCOL)
# print ('create dictionary')
# dictionary = corpora.Dictionary(corpus)

# dictionary.save('texts.dict') # store the dictionary, for future
# print (len(dictionary.token2id))
# print ('finish preprocessing data')
# print ('start training word2vec model using fastText')
# model = FastText(corpus, size = 50, min_count = 1, window = 2, sg = 1, hs = 0, iter = 10)
# fname = get_tmpfile("word2vec_ver7.model")
# model.save(fname)
# print ('finish train word2vec model')
# print ('start test word2vec')
# model = FastText.load(fname)
# existent_word = "đầu_tiên"
# # existent_word in model.wv.vocab
# vec = model.wv[existent_word]
# print (vec)