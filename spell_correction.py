from nltk.tokenize import sent_tokenize, word_tokenize
from spelling.data_cleaner import *
from spelling.edit_distance import *
from nltk import ngrams
import itertools
corpus_file = "./data/corpus.txt"
with open(corpus_file, encoding="utf-8") as f:
    corpus = f.read().lower()
corpus = clean_corpus(corpus)
print (corpus)
n = 2 # number of neighbor add to gram
ngram_arr, count_ngram_arr = count_ngram_corpus(corpus,n)
print (ngram_arr, count_ngram_arr)
def count_ngram(arr):
    if(arr in ngram_arr):
        ind = ngram_arr.index(arr)
        return count_ngram_arr[ind]
    else:
        return 0
    
def n_gram_correlated(arr1,arr2,arr3):
    results = []
    most_correlated = []
    max_correlated = -1
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            for k in range(len(arr3)):
                ngram = [arr1[i],arr2[j],arr3[k]]
                number_ngram = count_ngram(ngram)
                if (number_ngram > max_correlated):
                    most_correlated = ngram
                    max_correlated = number_ngram
    return most_correlated
def create_correlated_ngram(corelated_word):
    corr_ngram = []
    for i in range(len(corelated_word)):
        if(i==0):
            continue
        elif (i== len(corelated_word)-1 ):
            break
        else:
            corr_ngram.append(n_gram_correlated(corelated_word[i-1],corelated_word[i],corelated_word[i+1]))
    return corr_ngram
if __name__ == '__main__':
    sentence = 'ban ma chung khoan ssi'
    word_sentence = word_tokenize(sentence)
    corelated_sentence = [['#']]
    for word in word_sentence:
        corelated_sentence.append(check_editdistanceone(word))
    corelated_sentence.append(['#'])
    print (corelated_sentence)
    corr_ngram =  create_correlated_ngram(corelated_sentence)
    print (corr_ngram)
