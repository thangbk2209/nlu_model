import tensorflow as tf 
import pickle as pk 
import numpy as np 
from nltk.tokenize import sent_tokenize, word_tokenize
from pyvi import ViTokenizer,ViPosTagger
import re



def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return word2int, int2word
def to_one_hot(data_point_index,vocab_size):
    temp = np.zeros(vocab_size, dtype = np.int8)
    temp[data_point_index] = 1
    return temp
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
def compound_word(words):
    # this function transform output of bidirectional lstm tokenizer to format of pyvi
    tokens = []
    # print (words)
    # print (len(words))
    for i,word in enumerate(words):
        # print(word)
        token = ''
        if(word[1] == 'B_W'):
            token += word[0]
            for j in range(i+1,len(words),1):
                if(words[j][1] == 'I_W' ):
                    token = token + '_' + words[j][0]
                else:
                    break
            tokens.append(token)
        elif(word[1] == 'O'):
            tokens.append(word[0])
        else:
            continue
    # print (tokens)
    return tokens
def separate_word(tokens):
    # this function transform output of pyvi tokenizer to format of bidirectional lstm
    word_separate = []
    for token in tokens:
        words = token.split('_')
        if len(words) == 1:
            word_separate.append([words[0],'B_W'])
        else:
            arr = []
            word_separate.append([words[0],'B_W'])
            for i in range(1,len(words),1):
                word_separate.append([words[i],'I_W'])
    return word_separate



file_to_save_model = 'FinTokenizer/model_saved_ver12'
word2int, int2word = read_trained_data('./vocabulary/word2int_ver12.pkl')
corpus_file = './data/official_corpus.txt'
input_size = 64
num_units = [32,4]
embedding_dim = 50
epochs = 1
batch_size = 512
learning_rate = 0.2

def finTokenizer(text):
    all_tokens = []
    x_one_hot_vector = []
    real_word = []
    # file_word = open('word.txt','w', encoding="utf8")
    number_replace = '1000'
    number_words = len(word2int)
    all_single_word = word_tokenize(text)
    if(len(all_single_word) <= 64):
        real_word = all_single_word
    # else:
    
    for j in range(len(all_single_word)):
        if(hasNumbers(all_single_word[j])):
            x_one_hot_vector.append(to_one_hot(word2int[number_replace], number_words)) 
        else:
            if all_single_word[j] in word2int:
                x_one_hot_vector.append(to_one_hot(word2int[all_single_word[j]], number_words)) 
            else:
                x_one_hot_vector.append(to_one_hot(word2int[number_replace], number_words)) 
    for i in range(len(all_single_word), input_size, 1):
        temp = np.zeros(number_words,dtype = np.int8)
        x_one_hot_vector.append(temp)
    x_data = []
    x_data.append(x_one_hot_vector)
    x_data = np.asarray(x_data)
    # print (x_data.shape)
    # lol
    # print("preprocessing done")
    real_word = np.asarray(real_word)
    # print (real_word)
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["tag"] ,export_dir = file_to_save_model)
        # Access and create placeholders variables and
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("sentence_one_hot:0")
        y_label = graph.get_tensor_by_name("y_label:0") 
        prediction = graph.get_tensor_by_name("outputs/Softmax:0")
        pred = (sess.run(prediction,{x:x_data}))
        # print ("tokenize done")
        # print ("start showing results")
        all_index = tf.argmax(pred, axis=2, name=None)
        all_index = sess.run(all_index)
        labelsi = []
        # print (real_word)
        # print (all_index)
        for i in range(len(real_word)):
            if (all_index[0][i] == 0):
                labelsi.append([real_word[i],'B_W'])
            elif (all_index[0][i] == 1):
                labelsi.append([real_word[i],'I_W'])
            else :
                labelsi.append([real_word[i],'O'])
        tokens = compound_word(labelsi)
        all_tokens.append(tokens)
        all_tokens = np.asarray(all_tokens)
        # print (all_tokens.shape)
        sess.close()
    return all_tokens
def tokenize_corpus(sentences):
    # print (len(sentences))
    # lol
    # not_have = []
    all_tokens = []
    number_sentence = 0
    x_one_hot_vector = []
    real_word = []
    # file_word = open('word.txt','w', encoding="utf8")
    number_replace = '1000'
    number_words = len(word2int)
    num = 0
    for sentence in sentences:
        sentence = sentence[:-1]
        number_sentence+=1
        all_single_word = word_tokenize(sentence)
        if(len(all_single_word)>64):
            num+=1
            continue
        else:
            real_word.append(all_single_word)
        x_one_hot_vectori = []
        for j in range(len(all_single_word)):
            if(hasNumbers(all_single_word[j])):
                x_one_hot_vectori.append(to_one_hot(word2int[number_replace], number_words)) 
            else:
                if all_single_word[j] in word2int:
                    x_one_hot_vectori.append(to_one_hot(word2int[all_single_word[j]], number_words)) 
                else:
                    x_one_hot_vectori.append(to_one_hot(word2int[number_replace], number_words)) 
        for i in range(len(all_single_word), input_size, 1):
            temp = np.zeros(number_words,dtype = np.int8)
            x_one_hot_vectori.append(temp)
        x_one_hot_vector.append(x_one_hot_vectori)
    x_data = np.asarray(x_one_hot_vector)
    print("preprocessing done")
    print ("number of sentences that have length longer than 64:",num)
    real_word = np.asarray(real_word)
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["tag"] ,export_dir = file_to_save_model)
        # Access and create placeholders variables and
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("sentence_one_hot:0")
        y_label = graph.get_tensor_by_name("y_label:0") 
        prediction = graph.get_tensor_by_name("outputs/Softmax:0")
        pred = (sess.run(prediction,{x:x_data}))
        print ("tokenize done")
        print ("start showing results")
        all_index = tf.argmax(pred, axis=2, name=None)
        all_index = sess.run(all_index)
        for i in range(real_word.shape[0]):
            # print (i)
            labelsi = []
            for j in range(len(real_word[i])):
                if (all_index[i][j] == 0):
                    labelsi.append([real_word[i][j],'B_W'])
                elif (all_index[i][j] == 1):
                    labelsi.append([real_word[i][j],'I_W'])
                else :
                    labelsi.append([real_word[i][j],'O'])
            tokens = compound_word(labelsi)
            all_tokens.append(tokens)
        all_tokens = np.asarray(all_tokens)
        print (all_tokens.shape)
    return all_tokens
if __name__ == '__main__':
    # all_tokens = tokenize_corpus()
    # print (all_tokens)
    sentences = ["tôi muốn mua 100 cổ phiếu ssi giá 12.4."]
    all_tokens = finTokenizer(sentences[0])
    print (all_tokens)