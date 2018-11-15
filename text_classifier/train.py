from preprocessing_data import PreprocessingDataClassifier
from pyvi import ViTokenizer, ViPosTagger
from gensim import corpora, models, similarities
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid
import pickle as pk
from ANN_classifier import Classifier
queue = Queue()
# read data to a filel
CLASSIFY_BY_SOFTMAX = 1
CLASSIFY_BY_SVM = 2
CLASSIFY_BY_KNN = 3
CLASSIFY_BY_BAYESIAN = 4
OPTIMIZER_BY_GRADIENT = 5
OPTIMIZER_BY_SGD = 6
OPTIMIZER_BY_ADAM = 7
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        vectors = pk.load(input_file)
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return vectors, word2int, int2word
def train_model(batch_size_classifier):
    # batch_size_classifier = item["batch_size_classifier"]
    summary = open("evaluate.csv",'a+')
    
    for window_size in window_sizes:
        for embedding_dim in embedding_dims:
            for batch_size_word2vec in batch_size_word2vecs:
                version = str(version_wc) + '.'+str(version_classifier)
                file_to_save_classified_data = 'results/ANN_ver' + version + '/ws-' + '-embed-' + str(embedding_dim) + 'batch_size_cl' + str(batch_size_classifier)
                
                classifier = Classifier(input_size, num_classes, epoch_classifier ,embedding_dim,
                        batch_size_classifier, optimizer_method, wv_file, learning_rate,
                        file_to_save_classified_data=file_to_save_classified_data)
                print (file_data_classifier)
                accuracy, int2intent = classifier.train(file_data_classifier)
                summary.write(str(window_size)+","+str(embedding_dim)+","+str(batch_size_word2vec)+","+str(batch_size_classifier)+"," + version + ","+str(accuracy)+"\n")
                # print (int2intent)
window_sizes = [2]
embedding_dims = [50]
batch_size_word2vecs = [4]
file_to_save_word2vec_datas = []
input_size = 32
num_classes = 12
epoch_classifier = 1000

version_wc = 10
version_classifier = 11
learning_rate = 0.05
wv_file = "/home/fdm-thang/robochat/nlu_gensim/word_vector/word2vec_ver" + str(version_wc) + ".model"
file_data_classifier = '../data/text_classifier_ver15.txt'
optimizer_method = OPTIMIZER_BY_GRADIENT
batch_size_classifiers = [8]

# Consumer
if __name__ == '__main__':

    for i in range(len(batch_size_classifiers)):
        train_model(batch_size_classifiers[i])