import numpy,re,keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#load the text file

t= "humanAction.txt"
#t = "hello.txt"
text = open(t).read()
#to reduce the vocabulay lstm has to learn(only read lower case char)
text = text.lower()
text = text[:1136761/2]


#removing special charecters
pattern=re.compile("[^\w\n]")
text = pattern.sub(' ', text)

#mapping char to integers
chars = sorted(list(set(text)))
char_to_int = dict((c,i) for i,c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(text)
n_vocab = len(chars)
print n_chars , n_vocab

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = text[i:i + seq_length]
	seq_out = text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
#print "Total Patterns: ", n_patterns

#training data in format (sample, time step,features)
trainX = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize X
trainX = trainX / float(n_vocab)
trainY = np_utils.to_categorical(dataY)
#print trainY.shape,trainX.shape

import sys
# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(trainY.shape[1], activation='softmax'))
model.load_weights("weights-improvement-17-1.2324.hdf5")
#adamax= keras.optimizers.Adamax(lr=0.005)
model.compile(loss='categorical_crossentropy', optimizer='adamax',metrics=['accuracy'])

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.summary()
model.fit(trainX, trainY, nb_epoch=20,validation_split=0.40, batch_size=100,callbacks=callbacks_list)

start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# generate characters

text_file = open("output1.txt", "a")
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	text_file.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print "\nDone."