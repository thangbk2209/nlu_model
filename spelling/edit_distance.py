import pickle as pk 
import time
# Python program to check if given two strings are 
# at distance one 
  
# Returns true if edit distance between s1 and s2 is 
# one, else false 
def isEditDistanceOne(s1, s2): 
  
    # Find lengths of given strings 
    m = len(s1) 
    n = len(s2) 
  
    # If difference between lengths is more than 1, 
    # then strings can't be at one distance 
    if abs(m - n) > 1: 
        return False 
  
    count = 0    # Count of isEditDistanceOne 
  
    i = 0
    j = 0
    while i < m and j < n: 
        # If current characters dont match 
        if s1[i] != s2[j]: 
            if count == 1: 
                return False 
  
            # If length of one string is 
            # more, then only possible edit 
            # is to remove a character 
            if m > n: 
                i+=1
            elif m < n: 
                j+=1
            else:    # If lengths of both strings is same 
                i+=1
                j+=1
  
            # Increment count of edits 
            count+=1
  
        else:    # if current characters match 
            i+=1
            j+=1
  
    # if last character is extra in any string 
    if i < m or j < n: 
        count+=1
  
    return count == 1
def read_vocab_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return word2int, int2word
def read_tf_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        tf_dicts = pk.load(input_file)
    return tf_dicts
def get_vocab():
    word2int, int2word = read_vocab_data('./vocabulary/word2int_ver12.pkl')
    vocab = word2int.keys()
    return vocab
def check_editdistanceone(word):
    vocab = get_vocab()
    results = []
    for element in vocab:
        if (isEditDistanceOne(word,element) == True):
            results.append(element)
    results.append(word)
    return results

if __name__ == '__main__':
    start_time = time.time()
    word = 'mem'
    correlated_word = check_editdistanceone(word)
    print (correlated_word.shape)
    print (time.time() - start_time)