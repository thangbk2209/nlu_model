from preprocessing_data import PreprocessingDataClassifier
import tensorflow as tf 
import pickle as pk 
import numpy as np 
import nltk
from nltk.tokenize import sent_tokenize
from pyvi import ViTokenizer, ViPosTagger
from gensim import corpora, models, similarities
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from tokenizer import *
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        vectors = pk.load(input_file)
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return vectors, word2int, int2word

input_size = 32
embedding_dim = 100

fname = get_tmpfile("word2vec_ver4.model")
model = FastText.load(fname)

texts = []
intents_data = []
intents_official = ['end', 'trade', 'cash_balance', 'advice', 'order_status', 'stock_balance', 'market', 'cancel']
sentences = {}
intents_filter = intents_official
intents = list(intents_data)
intents_size = len(intents_filter)
sentence = "Đêm qua mơ các cụ bảo nên mua ssi, thị trường ssi hôm nay thế nào?"
def to_one_hot(index_of_intent,intent_size):
    temp = np.zeros(intent_size)
    temp[index_of_intent] = 1
    return list(temp)
intent2int = {}
int2intent = {}
x_train = []
y_train = []
all_sentences = []
for index,intent in enumerate(intents_official):
    intent2int[intent] = index
    int2intent[index] = intent 
    # print (i)
# all_words = ViPosTagger.postagging(ViTokenizer.tokenize(sentence))[0]
all_words = finTokenizer(sentence)[0]
print (all_words)
data_x_raw = []
# print (i)
# print (all_words)
for word in all_words:
    # print (word)
    data_x_raw.append(model.wv[word])
for k in range(input_size - len(data_x_raw)):
    padding = np.zeros(embedding_dim)
    data_x_raw.append(padding)
data_x_original = [data_x_raw]
data_x_original = np.asarray(data_x_original)
print (data_x_original.shape)
data_x_original = np.reshape(data_x_original,(data_x_original.shape[0],data_x_original.shape[1]*data_x_original.shape[2]))
# print (data_x_original.shape)
# lol
tf.reset_default_graph()
with tf.Session() as sess:
    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('results/ANN_ver3/ws--embed-100batch_size_cl8.meta')
    saver.restore(sess,tf.train.latest_checkpoint('results/ANN_ver3/'))
    # Access and create placeholders variables and
    graph = tf.get_default_graph()
    # print ([n.name for n in tf.get_default_graph().as_graph_def().node])
    x = graph.get_tensor_by_name("x:0")
    y_label = graph.get_tensor_by_name("y_label:0")
    # Access the op that you want to run. 
    prediction = graph.get_tensor_by_name("prediction/Softmax:0")
    # for i in range(epochs):
        # for j in range(total_batch):
            # batch_x_train, batch_y_train = x_train[j*batch_size_classifier:(j+1)*batch_size_classifier], y_train[j*batch_size_classifier:(j+1)*batch_size_classifier]
    pred = (sess.run(prediction,{x:data_x_original}))
    print (pred)
    corr_pred = tf.reduce_max(pred)
    index = tf.argmax(pred, axis=1, name=None)
    corr = sess.run(corr_pred)
    print (corr)
    ind = sess.run(index)[0]
    print (ind)
    print (intents_official[ind])
            # print (batch_x_train[0])
            # print (batch_y_train[0])
            # train_op = sess.graph.get_operation_by_name('training_step')
            # sess.run(train_op,{x:batch_x_train,y_label: batch_y_train})
            # print sess.run(pred)
            # print (sess.run(corr_pred))
    # print (sess.run(corr_pred))
    # print (sess.run(index))
        # print (sess.run(corr_pred)[i])
        # print (int2intent[ind[i]])
            # print ('epoch: ',i,'Done')
    # save_path = saver.save(sess, '../../results/text_classification/ANN_ver6/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size_w2c-' + str(batch_size_word2vec) + 'batch_size_cl8')
        # a = tf.maximum(pred)
        # print (sess.run(a))