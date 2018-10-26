import numpy as np
leng = 0
train_file = open('train.txt','w')
test_file = open('test.txt','w')

with open ("./text_classifier_ver8.txt", encoding="utf8") as input:
    for line in input:
        leng = leng + 1
print (leng)
test_size = int(leng * 0.2)
print (test_size)
test = np.random.choice(1538,test_size,replace=False)
test = sorted(list(test))
for i in range(leng):
    if i not in test:
        train_file.write(str(i)+" ")
    else:
        test_file.write(str(i)+" ")
train_file.close()
test_file.close()
with open('train.txt','r') as inp:
            line = inp.readline()
            line = line.strip()
            #for line in inp:
            temp = line.split(" ")
            h = [int(i) for i in temp]

print(h)