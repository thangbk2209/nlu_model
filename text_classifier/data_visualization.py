import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
font = {'size'   : 5}

matplotlib.rc('font', **font)

def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, no_movies)
    plt.xlabel('Genre', fontsize=5)
    plt.ylabel('No of Movies', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Market Share for Each Genre 1995-2017')
    plt.show()

intents_data = []
with open('../data/text_classifier_ver8.txt', encoding="utf8") as input:
    for line in input :
        # print (line)
        temp = line.split(",",1)
        intents_data.append(temp[0]) #list of label
print (intents_data)
import collections
x = collections.Counter(intents_data)
print (x.values())
print(x)
l = np.arange(len(x.keys()))
print (l)
plt.bar(l, x.values(), align='center')
plt.xticks(l, x.keys())
# tick.label.set_fontsize(14) 
# plt.show()
# plt.savefig('./results/visulization_data_ver8.png')
plt.hist(intents_data)
plt.title("Data visualization")
plt.xlabel("Intents")
plt.ylabel("Frequency")
plt.show()
# plt.savefig('./results/visulization_data_ver8.png')
# fig = plt.gcf()

# plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')