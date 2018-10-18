import nltk
from nltk.tokenize import sent_tokenize
from pyvi import ViTokenizer, ViPosTagger
from gensim import corpora, models, similarities
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from tokenizer import *
# sentences = ["tôi muốn mua 100 cổ phiếu ssi giá 12.4."]
# all_tokens = finTokenizer(sentences[0])
# print (all_tokens)
corpus_file = './data/corpus.txt'
with open(corpus_file, encoding="utf-8") as f:
    corpus = f.read().lower()
    print("----------------------------------CORPUS----")
sentences = sent_tokenize(corpus)
print (len(sentences))
print ('start preprocessing data')
tokenized_corpus = []
print ('start tokenize word using pyvi library')
number_of_words = 0
i = 0
for sentence in sentences:
    i+=1
    print (i)
    words = finTokenizer(sentence)
    tokenized_corpus.append(words[0])
    number_of_words += len(words[0])
print (number_of_words)
# print (tokenized_corpus)
print ('create dictionary')
dictionary = corpora.Dictionary(tokenized_corpus)
dictionary.save('texts.dict') # store the dictionary, for future
print (dictionary.token2id)
print ('finish preprocessing data')
print ('start training word2vec model using fastText')
model = FastText(tokenized_corpus, size = 50, min_count = 1, window = 2, sg = 1, hs = 0, iter = 10)
fname = get_tmpfile("word2vec_ver1.model")
model.save(fname)
print ('finish train word2vec model')
print ('test')
model = FastText.load(fname)
existent_word = "đầu_tiên"
existent_word in model.wv.vocab
vec = model.wv[existent_word]
print (vec)