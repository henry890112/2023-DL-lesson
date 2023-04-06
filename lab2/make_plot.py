# use matplotlib to make a plot of the data
import matplotlib.pyplot as plt
import numpy as np

# read in the csv file
data = np.genfromtxt('./csv/henry_mean_score1.csv', delimiter=',')
# print(data)
# get the index of the data
index = np.arange(data.shape[0])
# print(index)

# read in the csv file
data3 = np.genfromtxt('./csv/henry_mean_score3.csv', delimiter=',')
index3 = np.arange(data3.shape[0])

# read in the csv file
data5 = np.genfromtxt('./csv/henry_mean_score5.csv', delimiter=',')
index5 = np.arange(data5.shape[0])

# read in the csv file
data7 = np.genfromtxt('./csv/henry_mean_score7.csv', delimiter=',')
index7 = np.arange(data7.shape[0])

#read in the csv file
data9 = np.genfromtxt('henry_mean_score9.csv', delimiter=',')

# plot the data
plt.plot((index+1)* 1000, data, label='lr_0.1')
plt.plot((index3+1)* 1000, data3, label='lr_0.3')
plt.plot((index5+1)* 1000, data5, label='lr_0.5')
plt.plot((index7+1)* 1000, data7, label='lr_0.7')
plt.plot((index7+1)* 1000, data9, label='lr_0.9')

plt.xlabel('Epoch')
plt.ylabel('Mean Score')
plt.title('Mean Score vs Epoch')
plt.legend()

plt.savefig('henry_mean_score.png')

