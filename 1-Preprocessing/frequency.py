# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading csv file as pandas dataframe
df = pd.read_csv('0-Datasets/WineQT.data')
df.head()

df = df['quality'].value_counts()
print(df)

'''x1 = df.loc[df.quality=='3', 'total sulfur dioxide']
x2 = df.loc[df.quality=='4', 'total sulfur dioxide']
x3 = df.loc[df.quality=='5', 'total sulfur dioxide']
x4 = df.loc[df.quality=='6', 'total sulfur dioxide']
x5 = df.loc[df.quality=='7', 'total sulfur dioxide']
x6 = df.loc[df.quality=='8', 'total sulfur dioxide']

kwargs = dict(alpha=0.5, bins=100)

plt.hist(x1, **kwargs, color='g', label='3')
plt.hist(x2, **kwargs, color='b', label='4')
plt.hist(x3, **kwargs, color='r', label='5')
plt.hist(x4, **kwargs, color='y', label='6')
plt.hist(x5, **kwargs, color='indigo', label='7')
plt.hist(x6, **kwargs, color='gold', label='8')
plt.gca().set(title='Frequency Histogram of Quality Wines', ylabel='Frequency')
plt.xlim(50,75)
plt.legend();'''

notas=['3','4','5','6','7','8']
frequencia=[10,9,8,7,6,5]

plt.bar(notas, frequencia, color='red')
plt.xticks(notas)
plt.ylabel('Frequencia')
plt.xlabel('Nota')
plt.title('Histograma')
plt.show()