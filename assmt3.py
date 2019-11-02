# -*- coding: utf-8 -*-

from keras.preprocessing.sequence import pad_sequences
from gensim import models
from numpy import zeros
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation,TimeDistributed,InputLayer
from keras import optimizers
from keras.optimizers import Adam
from keras.layers import Embedding
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
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])
     
    print(model.summary())
    return model

if __name__ == "__main__":
    line_list=[]
    tag_list=[]
    line_list,tag_list,tag_set=openandread(r'D:\bin\AIT-726\Assignemnts\conll2003\train.txt',line_list,tag_list)
    
    max_len = len(max(line_list, key=len))
    
    line_vec,word2index=vectorize_line(line_list,vocab)
    line_vec = pad_seq(line_vec,max_len)
    
    embedding_matrix=embed(word2index)
    
    padded_tag = pad_tags(tag_list,max_len)
    tag_set.add('<pad>')
    tag_vec=vectorize_tag(tag_set,padded_tag)
    
    line_vec = pad_seq(line_vec,max_len)
    
    tag_vec_one_hot = to_categorical(tag_vec,11)
    
    model=vanilla_rnn(max_len,embedding_matrix)
    model.fit(line_vec, tag_vec_one_hot, batch_size=2000, epochs = 5, validation_split=0.2)
    
