import regex
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
def execute_number(element):
    if(element.isnumeric() == True):
        element = '1000'
    if (re.match("^\d+?\.\d+?$", element) is None):
        print ("Not float")
    else:
        element = '1000'
    return element
def clean_corpus(corpus):
    corpus = execute_special_character(corpus)
    all_sentences = sent_tokenize(corpus)
    all_sentence_words = []
    for sentence in all_sentences:
        word_sentence = word_tokenize(sentence)
        for i,word in enumerate(word_sentence):
            word_sentence[i] = execute_number(word)
        all_sentence_words.append(word_sentence)
    return all_sentence_words
def clean_sentence(sentence):
    sentence = execute_special_character(sentence)
    word_sentence = word_tokenize(sentence)
    return word_sentence
def n_gram_corpus(corpus,n):
    # corpus = execute_special_character(corpus)
    # all_sentences = sent_tokenize(corpus)
    all_sentence_ngram= []
    # print (corpus)
    for sentence in corpus:
        # word_sentence = word_tokenize(sentence)
        all_sentence_ngram.append(n_gram(sentence,n))
    return all_sentence_ngram
def count_ngram_corpus(corpus,n):
    ngram_arr = []
    count_ngram_arr = []
    all_sentence_ngram = n_gram_corpus(corpus,n)
    print (all_sentence_ngram)
    for i in range(len(all_sentence_ngram)):
        for j in range(len(all_sentence_ngram[i])):
            if(all_sentence_ngram[i][j] not in ngram_arr):
                ngram_arr.append(all_sentence_ngram[i][j])
                count_ngram_arr.append(1)
            else:
                ind = ngram_arr.index(all_sentence_ngram[i][j])
                count_ngram_arr[ind] += 1
    return ngram_arr, count_ngram_arr
def n_gram(sentence,n):
    results = []
    n_gram_array = ngrams(sentence, n)
    print (n_gram_array)
    grams = []
    for gram in (n_gram_array):
        grams.append(gram)
    # return (n_gram_array)
    print (len(grams))
    for i,gram in enumerate(grams):
        print (gram)
        if(i == 0):
            resultsi = ['#']
            resultsi.append(gram[0])
            resultsi.append(gram[1])
        # elif(i == len(grams) -1 ):
        #     resultsi = [gram[0]]
        #     resultsi.append(gram[1])
        #     resultsi.append('#')
        else:
            resultsi = [grams[i-1][0]]
            resultsi.append(grams[i-1][1])
            resultsi.append(gram[1])
        results.append(resultsi)
    results.append([grams[-1][0],grams[-1][1],'#'])
    return results
def execute_special_character(text):
    # use regular expression to replace special characer and acronym
    text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
    text = regex.sub("&[a-z]+;", "", text) # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
    text = regex.sub("(?s){.+?}", "", text) # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
    # text = regex.sub("\S*\d\S*", "", text)  # remove word contain string and number
    text = regex.sub("[0-9]{2}/[0-9]{2}/[0-9]{4}", "", text) # remove media links
    # text = regex.sub("[-+]?[0-9]*\.?[0-9]*", "", text)  # remove float number
    text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text) # remove bold symbols
    text = regex.sub("[']{2}", "", text) # remove italic symbols
    text = regex.sub("\'"," ",text)  # remove ' character
    text = regex.sub(","," ",text)  # remove , character
    text = regex.sub("\["," ",text)  # remove [ character
    text = regex.sub("\]"," ",text)  # remove [ character
    text = regex.sub('[@!#$%^&;*()<–>?/\"“”,|}{~:]',' ',text)
    # this file include financial symbols
    symbol_arr = []
    with open ('./data/stockslist.txt',encoding = 'utf-8') as acro_file:
        lines = acro_file.readlines()
        for line in lines:
            symboli = line.rstrip('\n').split(',')
            symbol_arr.append(symboli[0].lower())
    # this file include acronym words
    acronym_arr = []
    with open ('./data/acronym_3characters.txt',encoding = 'utf-8') as acro_file:
        lines = acro_file.readlines()
        for line in lines:
            acroi = line.rstrip('\n').split(',')
            acronym_arr.append(acroi)
    for i in range(len(acronym_arr)):
        text = re.sub(r'\A%s\s' % acronym_arr[i][0], r' %s ' % acronym_arr[i][1], text)
        text = re.sub(r'\s%s\Z' % acronym_arr[i][0], r' %s ' % acronym_arr[i][1], text)
        text = re.sub(r'\s%s\s' % acronym_arr[i][0], r' %s ' % acronym_arr[i][1], text)
        text = re.sub(r'\s%s\W' % acronym_arr[i][0], r' %s ' % acronym_arr[i][1], text)
    return text
if __name__ == '__main__':
    corpus_file = "./data/corpus.txt"
    with open(corpus_file, encoding="utf-8") as f:
        corpus = f.read().lower()
    # print (corpus)
    sentences = ["tôi muốn mua 100 cổ phiếu ssi giá 12.4."]
    word_sentence = clean_sentence(sentences[0])
    # check_number ('1.2')
    # print (word_sentence)
    # n_gram_arr = n_gram(word_sentence,2)
    # print (n_gram_arr)
    # for gram in bi_grams:
    #     print (gram)
    corpus = clean_corpus(corpus)
    # print (corpus)
    # all_sentence_ngram = n_gram_corpus(corpus,2)
    # print (all_sentence_ngram)
    ngram_arr, count_ngram_arr = count_ngram_corpus(corpus,2)
    print (ngram_arr,count_ngram_arr)