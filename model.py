#%%
import os
import pickle

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from bert import tokenization

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
tf.gfile = tf.io.gfile
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from tensorflow.keras.layers import Dense, Input 
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import tools
#%%

#%%
train = pd.read_csv('../datasets/csvs/train.csv')
test = pd.read_csv('../datasets/csvs/test.csv')
submit = pd.read_csv("../datasets/csvs/sample_submission.csv")
#%%

#%%
train.drop(['keyword'], axis = 1)
train.drop(['location'], axis = 1)
train_label = train.pop('target')

test.drop(['keyword'], axis = 1)
test.drop(['location'], axis = 1)
#%%

#%%
with open('csvs/train_clean.pkl', 'rb') as file:
    train_clean = pickle.load(file)
with open('csvs/test_clean.pkl', 'rb') as file:
    test_clean = pickle.load(file)
#%%

#%%
train_clean
#%%

#%%
test_clean
#%%

# target_error_ids = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
# train.at[train['id'].isin(target_error_ids), 'target'] = 0.0

# traintweet_clean = []
# for tweet in train['text']:
#     traintweet_clean.append(tools.full_clean(tweet))
    
# train.drop(['text'], axis = 1)
# train['text'] = traintweet_clean

# with open('train_clean.pkl', 'wb') as file:
    # pickle.dump(train, file)

# testtweet_clean = []
# for tweet in test['text']:
#     testtweet_clean.append(tools.full_clean(tweet))

# test.drop(['text'], axis = 1)
# test['text'] = testtweet_clean

# with open('test_clean.pkl', 'wb') as file:
#     pickle.dump(test, file)

#%%
def bert_encode(texts, tokenizer, max_len = 512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        # tokenizing input
        text = tokenizer.tokenize(text)
        
        # placing bert tokens
        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        # getting amount to pad input by 
        pad_len = max_len - len(input_sequence)
        
        # converting strings to numerical ids 
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        # padding
        tokens += [0] * pad_len
        # creating padding masks
        pad = [1] * len(input_sequence) + [0] * pad_len
        # creating segment ids
        segment_id = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad)
        all_segments.append(segment_id)
    
    # return as numpy array to be used as model input
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def create_model(bert_layer, max_len = 512):
    input_id = Input(shape = (max_len,), dtype = tf.int32, name = "input_id")
    input_mask = Input(shape = (max_len,), dtype = tf.int32, name = "input_mask")
    segment_id = Input(shape = (max_len,), dtype = tf.int32, name = "segment_id")

    _, sequence = bert_layer([input_id, input_mask, segment_id])
    out = sequence[:, 0, :]

    # output layer dense layer with 1 neuron
    out = Dense(1, activation = 'sigmoid')(out)
    
    # constructing model
    model = Model(inputs = [input_id, input_mask, segment_id], outputs = out)
    # compiling model
    model.compile(Adam(lr = 2e-6), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model
#%%

#%%
bert_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(bert_url, trainable=True)
#%%

#%%
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
#%%

#%%
train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
#%%

#%%
train_input
#%%

#%%
test_input
#%%

#%%
model = create_model(bert_layer, max_len = 160)
model.summary()
#%%

#%%
checkpoint_filepath = 'modelcheckpoints/cp.ckpt'

model_checkpoint_callback = ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    monitor = 'val_accuracy',
    mode = 'max',
    save_best_only = True)
#%%

#%%
# model.fit(train_input, train_label, validation_split = 0.2, epochs = 10, verbose = 1, callbacks=[model_checkpoint_callback], batch_size = 10)
model.load_weights(checkpoint_filepath)
#%%


#%%
prediction = model.predict(test_input)
#%%

#%%
prediction
#%%

#%%
test
#%%

#%%
test_clean
#%%

#%%
print(test['text'][1])
#%%

#%%
print(prediction[1])
#%%

#%%
submit['target'] = prediction.round().astype(int)
submit.to_csv('csvs/submission.csv', index = False)
#%%