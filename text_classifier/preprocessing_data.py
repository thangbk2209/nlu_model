#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyvi import ViTokenizer, ViPosTagger
import numpy as np 
import re
from tokenizer import *
from gensim import corpora, models, similarities
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from data_cleaner import DataCleaner
class PreprocessingDataClassifier:
    """ This class prepare data for text_classification phrase.
    first, read the training data from text file: file_data_classifier.
    then, tokenize sentence from data and create vector for training phrase.
    using input_size to padding vector zeros for every vector => input from 
    each sample will have the same size
    """
    def __init__(self, embedding_dim = None, input_size = None, wv_file = None, file_data_classifier =""):
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.wv_file = wv_file
        self.file_data_classifier = file_data_classifier

    def preprocessing_data_fastText(self):
        model = FastText.load(self.wv_file)
        texts = []
        intents_data = [] # danh sách intents trong bộ dữ liệu
        intents_official = ['end', 'trade', 'cash_balance', 'advice', 'order_status', 'stock_balance', 'market', 'cancel']
        sentences = {}
        with open(self.file_data_classifier, encoding="utf8") as input:
            for line in input :
                print (line)
                temp = line.split(",",1)
                temp[1] = temp[1].lower()
                texts.append(temp[1])  #list of train_word
                intents_data.append(temp[0]) #list of label
                sentences[temp[1]] = temp[0]
        intents_filter = intents_official
        intents = list(intents_data)
        intents_size = len(intents_filter)
        """
        create vector one hot for label(intent)
        """
        def to_one_hot(index_of_intent,intent_size):
            temp = np.zeros(intent_size)
            temp[index_of_intent] = 1
            return list(temp)
        intent2int = {}
        int2intent = {}
        
        x_train = []
        y_train = []
        all_sentences = []
        for index,intent in enumerate(intents_filter):
            intent2int[intent] = index
            int2intent[index] = intent 
        all_sentences_word = tokenize_corpus(texts)
        data_cleaner = DataCleaner(all_sentences_word)
        all_sentences_word = data_cleaner.clean()
        for i, all_words in enumerate(all_sentences_word):
            data_x_raw = []
            for word in all_words:
                print ("word",word)
                data_x_raw.append(model.wv[word])
            for k in range(self.input_size - len(data_x_raw)):
                padding = np.zeros(self.embedding_dim)
                data_x_raw.append(padding)
            data_x_original = data_x_raw
            label = to_one_hot(intent2int[intents[i]], intents_size)

            x_train.append(data_x_original)
            y_train.append(label)
            all_sentences.append(all_words)
        data_classifier_size = len(x_train)
        with open('../data/train.txt') as input:
            line = input.readline()
            line = line.strip()
            temp = line.split(" ")
            train_index = [int(i) for i in temp]
        test_label = []
        train_label = []
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in train_index:
            train_x.append(x_train[i])
            train_y.append(y_train[i])
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        for i in range(data_classifier_size):
            
            if i not in train_index:
                test_label.append(intents[i])
                test_x.append(x_train[i])
                test_y.append(y_train[i])
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)
        
        return train_x, train_y, test_x, test_y, int2intent, test_label, all_sentences, texts
    def preprocessing_data(self):
        # stop_words = StopWord()
        texts = []
        intents_data = [] # danh sách intents trong bộ dữ liệu
        intents_official = ['end', 'trade', 'cash_balance', 'advice', 'order_status', 'stock_balance', 'market', 'cancel']
        sentences = {}
        with open(self.file_data_classifier, encoding="utf8") as input:
            for line in input :
                # print (line)
                temp = line.split(",",1)
                temp[1] = temp[1].lower()
                texts.append(temp[1])  #list of train_word
                intents_data.append(temp[0]) #list of label
                sentences[temp[1]] = temp[0]
        intents_filter = intents_official
        intents = list(intents_data)
        intents_size = len(intents_filter)
        # print (intents)
        """
        create vector one hot for label(intent)
        """
        def to_one_hot(index_of_intent,intent_size):
            temp = np.zeros(intent_size)
            temp[index_of_intent] = 1
            return list(temp)
        intent2int = {}
        int2intent = {}
        x_train1 = []
        x_train = []
        y_train = []
        all_sentences = []
        for index,intent in enumerate(intents_filter):
            intent2int[intent] = index
            int2intent[index] = intent 
        # print (int2intent)
        # lol
        for i, sentence in enumerate(texts):
            # print (i)
            data_cleaner = DataCleaner(sentence)
            all_words = data_cleaner.separate_sentence()
            data_x_raw = []
            # print (i)
            # print (all_words)
            for word in all_words:
                #print ("word",word)
                # print(self.vectors[self.word2int[word]])
                data_x_raw.append(self.vectors[self.word2int[word]])
            for k in range(self.input_size - len(data_x_raw)):
                padding = np.zeros(self.embedding_dim)
                data_x_raw.append(padding)
            data_x_original = data_x_raw
            # print(data_x_original)
            # data_x_original = np.sum(data_x_raw,axis = 0)
            # print (data_x_original)
            # lol
            label = to_one_hot(intent2int[intents[i]], intents_size)

            x_train.append(data_x_original)
            x_train1.append(np.average(data_x_original,axis = 0))
            # print (x_train1)
            # lol
            y_train.append(label)
            all_sentences.append(all_words)
        data_classifier_size = len(x_train)
        train_size = int(data_classifier_size * 0.8)
        with open('../../data/train/train.txt') as input:
            line = input.readline()
            line = line.strip()
            temp = line.split(" ")
            train_index = [int(i) for i in temp]
           # print(train_index)
        test_label = []
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        # train_x = x_train
        # train_y = y_train 
        for i in train_index:
            train_x.append(x_train1[i])
            train_y.append(y_train[i])
            #print("y_train",y_train[i])
           #print("x_train",y_train[i])
            
        for i in range(data_classifier_size):
            # print (i)
            if i not in train_index:
                test_label.append(intents[i])
                test_x.append(x_train1[i])
                test_y.append(y_train[i])
        # for i in range(data_classifier_size):
        #     test_label.append(intents[i])
        #     test_x.append(x_train[i])
        #     test_y.append(y_train[i])
                # print (i)
       #  train_x = x_train[i for i in train_index]
       # train_y = y_train[i for i in train_index]
       # test_x = x_train[i for i not in train_index]
       # test_y = y_train[i for i not in train_index ]
        
        return train_x, train_y, test_x, test_y, int2intent,test_label, all_sentences, texts
    def preprocessing_data_KNN(self):
        # stop_words = StopWord()
        texts = []
        intents_data = [] # danh sách intents trong bộ dữ liệu
        intents_official = ['end', 'trade', 'cash_balance', 'advice', 'order_status', 'stock_balance', 'market', 'cancel']
        sentences = {}
        with open(self.file_data_classifier, encoding="utf8") as input:
            for line in input :
                # print (line)
                temp = line.split(",",1)
                temp[1] = temp[1].lower()
                texts.append(temp[1])  #list of train_word
                intents_data.append(temp[0]) #list of label
                sentences[temp[1]] = temp[0]
        intents_filter = intents_official
        intents = list(intents_data)
        
        x_train = []
        y_train = []
        all_sentences = []
        for i, sentence in enumerate(texts):
            # print (i)
            data_cleaner = DataCleaner(sentence)
            all_words = data_cleaner.separate_sentence()
            data_x_raw = []
            # print (i)
            # print (all_words)
            for word in all_words:
                # print ('===============word=================')
                # print (self.vectors)
                data_x_raw.append(self.vectors[self.word2int[word]])
            for k in range(self.input_size - len(data_x_raw)):
                padding = np.zeros(self.embedding_dim)
                data_x_raw.append(padding)
            data_x_original = data_x_raw
            label = intents[i]
            x_train.append(data_x_original)
            
            y_train.append(label)
            all_sentences.append(all_words)
        x_train = np.array(x_train)
        x_train1 = np.array(x_train1)
        # print(x_train1.shape())
        # print (x_train1[:5])
        x_train = np.reshape(x_train,(len(x_train), self.input_size*self.embedding_dim))
        # print(x_train.shape())
        data_classifier_size = len(x_train)
        train_size = int(data_classifier_size * 0.8)
        with open('../../data/train/train.txt') as input:
            line = input.readline()
            line = line.strip()
            temp = line.split(" ")
            train_index = [int(i) for i in temp]
           # print(train_index)
        test_label = []
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        # train_x = x_train
        # train_y = y_train 
        for i in train_index:
            train_x.append(x_train[i])
            train_y.append(y_train[i])
            
        for i in range(data_classifier_size):
            # print (i)
            if i not in train_index:
                test_label.append(intents[i])
                test_x.append(x_train[i])
                test_y.append(y_train[i])
        
        return train_x, train_y, test_x, test_y, test_label, all_sentences, texts
