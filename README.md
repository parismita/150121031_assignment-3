# 150121031_assignment-3

used keras in theano

text prediction using lstm

the vocabulary is reduced from 93 char to 38 char using regex

hence it learns only numbers, alphabets and space and newline char

gives 64% aprox accuracy in 30-40 epoch

weights stored after every epoch as per epoch time taken is very large(used GPU +cuDNN)

text file generated : output1.txt

weights : weights-improvement-17-1.2324.hdf5

code: lstm.py
