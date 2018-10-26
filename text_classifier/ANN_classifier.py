import numpy as np
# from word2vec import Word2vec
import tensorflow as tf
from preprocessing_data import PreprocessingDataClassifier
import pickle
import matplotlib.pyplot as plt
import time
"""
this class training word2vec and receive a vector embedding and 
use this vector to train text classification by africial neural network model
"""
class Classifier:
    CLASSIFY_BY_SOFTMAX = 1
    CLASSIFY_BY_SVM = 2
    CLASSIFY_BY_KNN = 3
    CLASSIFY_BY_BAYESIAN = 4
    OPTIMIZER_BY_GRADIENT = 5
    OPTIMIZER_BY_SGD = 6
    OPTIMIZER_BY_ADAM = 7
    """
    This initial function define:
        input_size: The max size of a sentence, if the sentence have less word than
                    input_size, the model will add zero vector to sentence
        window_size: number of word in the left and right of current word to 
                    train word2vec
        epoch_word2vec, epoch_classifier: Number of epoch for train word2vec and text classifier
        embedding_dim: size of vector representation for each word
        num_classes : number of class(label) for training text classification
        file_to_save_classified_data : path of file to save vector presentation
    """
    def __init__(self, input_size = None, num_classes = None, 
     epoch_classifier = None ,embedding_dim = None,
    batch_size_classifier = None, optimizer_method = None, wv_file = None, learning_rate = 0.1, file_to_save_classified_data=""): 
        self.input_size = input_size
        self.epoch_classifier = epoch_classifier
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.file_to_save_classified_data = file_to_save_classified_data
        self.batch_size_classifier = batch_size_classifier
        self.optimizer_method = optimizer_method
        self.wv_file = wv_file
        self.learning_rate = learning_rate
        # print(self.file_to_save_classified_data)
    def early_stopping(self, array, patience):
        value = array[len(array) - patience - 1]
        arr = array[len(array)-patience:]
        check = 0
        for val in arr:
            if(val > value):
                check += 1
        if(check == patience):
            return False
        else:
            return True
    def classify(self, file_data_classifier):
        # Preprocessing data
        print ('------------------start preprocessing data ------------------')
        print (file_data_classifier)
        preprocessing_data = PreprocessingDataClassifier(self.embedding_dim, self.input_size, self.wv_file, file_data_classifier)
        print ('----------------------start training -----------------------')
        self.x_train, self.y_train, self.x_test, self.y_test, self.int2intent, self.test_label, self.all_sentences, self.texts = preprocessing_data.preprocessing_data_fastText()
        self.x_train = np.reshape(self.x_train,(self.x_train.shape[0],self.x_train.shape[1] * self.x_train.shape[2]))
        self.x_test = np.reshape(self.x_test,(self.x_test.shape[0],self.x_test.shape[1] * self.x_test.shape[2]))

        # Create graph
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, name="x", shape=(None, self.input_size * self.embedding_dim))
        hidden_value1 = tf.layers.dense(x, 2048, activation = tf.nn.relu, name="hidden1")
        hidden_value2 = tf.layers.dense(hidden_value1, 32, activation = tf.nn.sigmoid, name="hidden2")
        prediction = tf.layers.dense(hidden_value1,self.num_classes, activation = tf.nn.softmax, name="prediction")
        y_label = tf.placeholder(tf.float32, name="y_label", shape=(None, self.num_classes))
        # define the loss function:
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
        
        #select optimizer method 
        if self.optimizer_method == self.OPTIMIZER_BY_GRADIENT:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy_loss,name='training_step')
        elif self.optimizer_method == self.OPTIMIZER_BY_SGD:
            a = 0
        elif self.optimizer_method == self.OPTIMIZER_BY_ADAM:
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy_loss,name='training_step')
        sess = tf.Session()
        init = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        sess.run(init) #make sure you do this!
        # train for n_iter iterations
        total_batch = int(len(self.x_train)/ self.batch_size_classifier)
        loss_set = []
        for _ in range(self.epoch_classifier):
            start_time = time.time()
            avg_loss = 0
            for j in range(total_batch):
                batch_x_train, batch_y_train = self.x_train[j*self.batch_size_classifier:(j+1)*self.batch_size_classifier], self.y_train[j*self.batch_size_classifier:(j+1)*self.batch_size_classifier]
                sess.run(optimizer, feed_dict={x: batch_x_train, y_label: batch_y_train})
                loss = sess.run(cross_entropy_loss, feed_dict={x: batch_x_train, y_label: batch_y_train})/total_batch
                avg_loss += loss
            loss_set.append(avg_loss)
            print ('epoch: ', _ + 1)
            print('loss is : ',avg_loss)
            
            # if (_ > 20):
            #     if (self.early_stopping(loss_set, 20) == False):
            #         print ("early stopping training")
            #         break
            print("finished training classification phrase!!!")
            print ('time for epoch : ', _ + 1 , time.time()-start_time)

        print (self.x_test.shape)

        prediction = sess.run(prediction, feed_dict={x: self.x_test})
        save_path = saver.save(sess, self.file_to_save_classified_data)
        plt.plot(loss_set)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss'], loc='upper left')
        # plt.show()
        # plt.savefig('/home/thangnguyen/hust/lab/machine_learning_handling/history/history_mem.png')
        plt.savefig('training process.png')
        return prediction
    def save_trained_classifier_data(self,data):
        with open(self.file_to_save_classified_data,'wb+') as out:
            pickle.dump(data,out,pickle.HIGHEST_PROTOCOL)
    def train(self, file_data_classifier):
        print ('-------------------file_data_classifier----------------')
        print (file_data_classifier)
        prediction = self.classify(file_data_classifier)
        predict = []
        for i in range(len(prediction)):
            predict.append(self.int2intent[np.argmax(prediction[i])])
        correct = 0
        # fail_file = open('../../results/text_classification/fail.txt','w',encoding="utf8")
        with open('../data/train.txt') as input:
            line = input.readline()
            line = line.strip()
            temp = line.split(" ")
            train_index = [int(i) for i in temp]
            y = []
        for i in range(self.x_train.shape[0] + self.x_test.shape[0]):
            # print (i)
            if i not in train_index:
                y.append(i)
            # else:
            #     y_train.append(i)
        uncorrect = 0
        for i in range(len(predict)):
            if(predict[i] == self.test_label[i]):
                correct +=1
            else:
                uncorrect+=1
                print (uncorrect,i)
                print(self.test_label[i]+ ',' + predict[i] + ',' + self.texts[y[i]])
        accuracy = correct/len(self.test_label)
        print ('correct: ',correct)
        print ('test_label: ',len(self.test_label))
        print ("accuracy: ", accuracy)
        return accuracy, self.int2intent