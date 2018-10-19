"""
This file create others label for training classifier
"""
from nltk import sent_tokenize, word_tokenize
data_file = 'vn-news.txt'
with open(data_file, encoding="utf-8") as f:
    data = f.read().lower()
    print("----------------------------------CORPUS----")
sentences = sent_tokenize(data)
number_of_sentences = len(sentences)
print (number_of_sentences)

other_label_file = open('others.txt','w')

for sentence in sentences:
    words =word_tokenize(sentence)
    if(len(words)<=32):
        other_label_file.write("others," + sentence + "\n")
other_label_file.close()
