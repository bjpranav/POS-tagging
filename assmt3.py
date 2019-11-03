# -*- coding: utf-8 -*-

from keras.preprocessing.sequence import pad_sequences
from gensim import models
from numpy import zeros
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation,TimeDistributed,InputLayer,Bidirectional
from keras import optimizers
from keras.optimizers import Adam
from keras.layers import Embedding
from keras import backend as K
"""
Created on Sat Oct 26 17:28:47 2019

@author: Pranav Krishna
"""



def pad_tags(tags_list,max_len):
    #Pad the tags
    for i in range(0,len(tags_list)):
        tags_list[i] += ['<pad>'] * (max_len - len(tags_list[i]))
    print(tags_list[0])
    return tags_list
     
def pad_seq(line_list,max_len):
    #Pad each line
    sent_list = pad_sequences(line_list, maxlen=max_len)
    return sent_list

def normalize_case(s):    
    '''
    Paramaeter: Word to be normalized
    Converts words with capitalized first letters in to lower case.
    '''
    if(not s.isupper()):
        return s.lower()
    else:
        return s
    
vocab=set([])
def openandread(path,sent_list,tag_list):
    #load the file and create list for sentences and tags
    with open(path) as f:
        sent=[]
        tag=[]
        tag_set=set([])
        line_list=[]
        for line in f:
            content=line.split()
            if(line in ['\n', '\r\n']):
                line_list.append(sent)
                tag_list.append(tag)
                sent=[]
                tag=[]
            else:
                token=normalize_case(content[0])
                sent.append(token)
                vocab.add(token)
                tag.append(content[3])
                tag_set.add(content[3])
    return line_list,tag_list,tag_set


def vectorize_line(line_list,words):
    #Convert word to index
    word2index = {w: i for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs
    train_list=[]
    for s in line_list:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
        train_list.append(s_int)
    #embeddings_index = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #line_vec = [[w[word] for word in line if word in w] for line in line_list]
    return train_list,word2index
    
def vectorize_tag(tags,tag_list):
    #Convert tags to index
    tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
    test_tags_y=[]
    tag2index['<pad>']=0
    for s in tag_list:
        test_tags_y.append([tag2index[t] for t in s])
    return test_tags_y

def embed(word2index):
    #Create embedding matrix using word2vec
    embeddings_index = models.KeyedVectors.load_word2vec_format(r'D:\bin\AIT-726\Assignemnts\conll2003\GoogleNews-vectors-negative300.bin', binary=True)
    embedding_matrix = np.zeros((len(word2index)+2, 300))
    embeddings_index['-DOCSTART-']=np.zeros(300)
    for word, i in word2index.items():
        if(word in embeddings_index):
            embedding_vector = embeddings_index[word]
        else:
            embedding_vector =np.zeros(300)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    return embedding_matrix


def to_categorical(sequences, categories):
    #Create onehot encoding of y variable(tags)
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

def vanilla_rnn(max_len,embedding_matrix):
    #Create vannila rnn model
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len, )))
    model.add(Embedding(len(vocab), 128))
    model.add(SimpleRNN(256, return_sequences=True))
    model.add(TimeDistributed(Dense(11)))
    model.add(Activation('softmax'))
     
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.0001),
                  metrics=['accuracy'])
    print(model.summary())
    return model


def Bidirectional_rnn(max_len,embedding_matrix):
    #Create vannila rnn model
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len, )))
    model.add(Embedding(len(vocab), 128))
    model.add(Bidirectional(SimpleRNN(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(11)))
    model.add(Activation('softmax'))
     
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.0001),
                  metrics=['accuracy'])
    print(model.summary())
    return model

if __name__ == "__main__":
    line_list_train=[]
    tag_list_train=[]
    line_list_train,tag_list_train,tag_set_train=openandread(r'D:\bin\AIT-726\Assignemnts\conll2003\train.txt',line_list_train,tag_list_train)
    
    line_list_valid=[]
    tag_list_valid=[]
    line_list_valid,tag_list_valid,tag_set_valid=openandread(r'D:\bin\AIT-726\Assignemnts\conll2003\valid.txt',line_list_valid,tag_list_valid)
    
    line_list_test=[]
    tag_list_test=[]
    line_list_test,tag_list_test,tag_set_test=openandread(r'D:\bin\AIT-726\Assignemnts\conll2003\test.txt',line_list_test,tag_list_test)
    
    max_len_train = len(max(line_list_train, key=len))
    line_vec_train,word2index=vectorize_line(line_list_train,vocab)
    line_vec_train = pad_seq(line_vec_train,max_len_train)
    
    max_len_valid = len(max(line_list_valid, key=len))
    line_vec_valid,word2index=vectorize_line(line_list_valid,vocab)
    line_vec_valid = pad_seq(line_vec_valid,max_len_train)
    
    max_len_test = len(max(line_list_train, key=len))
    line_vec_test,word2index=vectorize_line(line_list_test,vocab)
    line_vec_test = pad_seq(line_vec_test,max_len_train)
    
    embedding_matrix=embed(word2index)
    
    padded_tag_train = pad_tags(tag_list_train,max_len_train)
    tag_set_train.add('<pad>')
    tag_vec_train=vectorize_tag(tag_set_train,padded_tag_train)
    
    padded_tag_valid = pad_tags(tag_list_valid,max_len_train)
    tag_set_train.add('<pad>')
    tag_vec_valid=vectorize_tag(tag_set_valid,padded_tag_valid)
    
    padded_tag_test = pad_tags(tag_list_test,max_len_train)
    tag_set_test.add('<pad>')
    tag_vec_test=vectorize_tag(tag_set_train,padded_tag_test)
    
    #line_vec_train = pad_seq(line_vec_train,max_len_train)
    #line_vec_valid = pad_seq(line_vec_valid,max_len_valid)
    #line_vec_test = pad_seq(line_vec_test,max_len_test)
    
    tag_vec_one_hot_train = to_categorical(tag_vec_train,11)
    tag_vec_one_hot_valid = to_categorical(tag_vec_valid,11)
    tag_vec_one_hot_test = to_categorical(tag_vec_test,11)
    
    #model=vanilla_rnn(max_len_train,embedding_matrix)
    #model.fit(line_vec_train, tag_vec_one_hot_train, batch_size=2000, epochs = 5,  validation_data=(line_vec_valid, tag_vec_one_hot_valid))
    
    
    model=Bidirectional_rnn(max_len_train,embedding_matrix)
    model.fit(line_vec_train, tag_vec_one_hot_train, batch_size=2000, epochs = 5,  validation_data=(line_vec_valid, tag_vec_one_hot_valid))
