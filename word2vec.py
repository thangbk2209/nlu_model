import nltk
from nltk.tokenize import sent_tokenize
from pyvi import ViTokenizer, ViPosTagger
from gensim import corpora, models, similarities
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from tokenizer import *
import math
# sentences = ["tôi muốn mua 100 cổ phiếu ssi giá 12.4."]
# all_tokens = finTokenizer(sentences[0])
# print (all_tokens)
corpus_file = './data/official_corpus.txt'
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
    # i+=1
    # print (sentences[0])
    # print (i*interval, (i+i)*interval)
    # print (sentences[i * interval: (i+1)*interval])
    texts = sentences[i * interval: (i+1)*interval]
    # print (texts)
    words = tokenize_corpus(texts)
    print (words.shape)
    for i in range(len(words)):
        tokenized_corpus.append(words[i])
tokenized_corpus = np.asarray(tokenized_corpus)
print (tokenized_corpus.shape)
# print (number_of_words)
print (tokenized_corpus[0:20])
print (tokenized_corpus[-20:-1])
print ('create dictionary')
dictionary = corpora.Dictionary(tokenized_corpus)
dictionary.save('texts.dict') # store the dictionary, for future
print (len(dictionary.token2id))
print ('finish preprocessing data')
print ('start training word2vec model using fastText')
model = FastText(tokenized_corpus, size = 100, min_count = 1, window = 2, sg = 1, hs = 0, iter = 10)
fname = get_tmpfile("word2vec_ver3.model")
model.save(fname)
print ('finish train word2vec model')
print ('start test word2vec')
model = FastText.load(fname)
existent_word = "đầu_tiên"
# existent_word in model.wv.vocab
vec = model.wv[existent_word]
print (vec)