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
from data_cleaner import DataCleaner
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        tf_dicts = pk.load(input_file)
    return tf_dicts

input_size = 32
embedding_dim = 50
version_wc = 10

version_classifier = 10
name_class = 'ANN_ver' + str(version_wc) + '.' + str(version_classifier)
wv_file = "/home/fdm-thang/robochat/nlu_gensim/word_vector/word2vec_ver" + str(version_wc) + ".model"
# fname = get_tmpfile("word2vec_ver9.model")
model = FastText.load(wv_file)
dict_file = "/home/fdm-thang/robochat/nlu_gensim/term_frequency/tf_ver" + str(version_wc) + ".pkl"
tf_dicts = read_trained_data(dict_file)
# print (tf_dicts)
# print (tf_dicts['nên'])

texts = []
intents_data = []
intents_official = ['end', 'trade', 'cash_balance', 'advice', 'order_status', 'stock_balance', 'market', 'cancel','ratio_status','look','name','cancel_all']
sentences = {}
intents_filter = intents_official
intents = list(intents_data)
intents_size = len(intents_filter)
def test(sentence):
    sentence = sentence.lower()
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
    all_words = finTokenizer(sentence)
    # print (all_words)
    # print (all_words)
    # all_sentences_word = tokenize_corpus(all_words)
    data_cleaner = DataCleaner(all_words)
    all_words = data_cleaner.clean()[0]
    # print (all_words)
    # print (all_words)
    data_x_raw = []
    # print (i)
    # print (all_words)
    num = 0
    num_stop_word = 0
    for word in all_words:
        # print (word)
        if ( word not in tf_dicts ):
            # print (word,tf_dicts[word])
            # print ('check',word)
            num+=1
        elif(data_cleaner.is_stop_word(word)):
            num_stop_word+=1
            # print ('stop word',word)
            continue
        else:
            # print (word,tf_dicts[word])
            data_x_raw.append(model.wv[word])
    # print (num)
    # print (len(all_words))
    # print (len(all_words)-num_stop_word)
    if (num >= 0.5*(len(all_words)-num_stop_word)):
        label = 'unknown label'
        return label
    else:
        # for k in range(input_size - len(data_x_raw)):
        #     padding = np.zeros(embedding_dim)
        #     data_x_raw.append(padding)
        data_x_original = [data_x_raw]
        data_x_original = np.asarray(data_x_original)
        data_x_original = np.average(data_x_original,axis = 1)
        # print (data_x_original.shape)
        # print (data_x_original.shape)
        # data_x_original = np.reshape(data_x_original,(data_x_original.shape[0],data_x_original.shape[1]*data_x_original.shape[2]))
        # print (data_x_original.shape)
        # lol
        tf.reset_default_graph()
        with tf.Session() as sess:
            
            #First let's load meta graph and restore weights
            saver = tf.train.import_meta_graph('results/'+ name_class + '/ws--embed-50batch_size_cl8.meta')
            saver.restore(sess,tf.train.latest_checkpoint('results/'+ name_class+'/'))
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
            # print (pred)
            corr_pred = tf.reduce_max(pred)
            index = tf.argmax(pred, axis=1, name=None)
            corr = sess.run(corr_pred)
            # print (corr)
            ind = sess.run(index)[0]
            # print (ind)
            return intents_official[ind]
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
# mua 100 ssi cho tôi khi mà thằng bạn xàm lìn của tôi bảo tôi nên thế 
# có người khuyên nên mua ssi, mua cho tôi ssi đi
# hôm qua thằng bạn tôi mua ssi giá 18, có nên mua ssi lúc này không nhỉ?
# Hôm qua thị trường bán tháo ssi, có nên mua vào lúc này không?
# Hôm qua tài khoản tiền của tôi về 15 triệu, cho tôi xem số dư cổ phiếu SSI trong tài khoản
# Cổ phiếu SSI về chưa nhỉ
# Tiền tôi bán cổ phiếu SSI về chưa nhỉ
# nên mua ssi không khi thằng bạn tôi mua ssi giá 18
# nên mua ssi không khi thằng bạn tôi mua ssi
# nên mua ssi không khi thằng bạn tôi mua ssi hôm qua
# Giờ phải xuống ăn cơm, đóng ứng dụng cho tôi
# Hôm nay bán SSI lỗ quá, dừng ứng dụng cho tôi
# bán SSI lỗ quá, dừng ứng dụng cho tôi
# hôm qua vừa mua 100 cổ phiếu ssi, xem lệnh mua ssi của tôi đã được khớp chưa
# sentence = "có nên sở hữu ssi lúc này không khi thi trường đang con gấu"
file = open('err_data.txt','w', encoding="utf8")
file1 = open('new_data.txt','w', encoding="utf8")

    
with open('/home/fdm-thang/robochat/nlu_gensim/data/testfile.txt') as input:   
    labels = []   
    contents = []
    for line in input :
        print (line)
        temp = line.split(",",1)
        labels.append(temp[0])
        contents.append(temp[1])
# print (len(contents))

all_sentences_words = []
# arr = []
# for i,content in enumerate(contents):
#     print (i)
#     all_words = finTokenizer(content)
#     # print (all_words)
#     # print (all_sentences_words)
#     data_cleaner = DataCleaner(all_words)
#     all_words = data_cleaner.clean()[0]
#     # print (all_words)
#     if(len(all_sentences_words) == 0):
#         all_sentences_words.append(all_words)
#         file1.write(labels[i] + ',' + content)
#         continue
#     elif (all_words not in all_sentences_words):
#         all_sentences_words.append(all_words)
#         file1.write(labels[i] + ',' + content)
#         continue
#     else:
#         pred_index = all_sentences_words.index(all_words)
#         all_sentences_words.append(all_words)
#         arr.append([pred_index+1,i+1,content])
#         # print (pred_index+1,i+1,content)
#         err = str(pred_index+1) +'\t' + str(i+1) +'\t' + str(content) + '\n'
#         file.write(err) 
# for i in range(len(arr)):
#     print (arr[i])
correct = 0 
for i,content in enumerate(contents):
    # print (i)
    label_predict = test(content)
    if (label_predict == labels[i]):
        # print (content,'---',labels[i])
        # print (label_predict)
        correct+=1
        # print (correct, i)
    else:
        print ("wrong", content,'---',labels[i],label_predict)
        # print (correct, i)
print (correct)

# label_predict = test("quay trở về chức năng chính")
# print (label_predict)