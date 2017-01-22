from __future__ import division
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
import tensorflow as tf
import pickle
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

tf.app.flags.DEFINE_boolean("decode",False)

FLAGS = tf.app.flags.FLAGS



pickle_in = open('data_set.pickle','rb')
train_x,train_y,test_x,test_y,weight,seq_length,n_sents,vocb_size,rev_vocab = pickle.load(pickle_in)



batch_size = 1
vocab_size =32

embedding_dim = 10

memory_dim = 10
num_layers=1





enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
           for t in range(seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,),
                        name="labels%i" % t)
          for t in range(seq_length)]

weights= [tf.placeholder(tf.float32, shape=(None,),
                        name="labels%i" % t)
          for t in range(seq_length)]



dec_inp = ([tf.zeros_like(labels[0], dtype=np.int32, name="GO")]
           + labels[:-1])
           


prev_mem = tf.zeros((batch_size, memory_dim))

single_cell = rnn_cell.BasicLSTMCell(memory_dim, state_is_tuple=True)

cell=single_cell
if num_layers > 1:
   cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)

with tf.variable_scope("decoders") as scope:

     dec_outputs, dec_memory = seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size,embedding_dim,feed_previous=False)
     
     scope.reuse_variables()

     decode_outputs_test, decode_state_test = seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp,cell,vocab_size,vocab_size,embedding_dim,feed_previous=True)








loss = seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)

tf.scalar_summary("loss", loss)



summary_op = tf.merge_all_summaries()

learning_rate = 0.5
momentum=0.9
optimizer = tf.train.MomentumOptimizer(learning_rate,momentum)
train_op = optimizer.minimize(loss)

saver = tf.train.Saver()


def train():
     with tf.Session() as sess:
         sess.run(tf.initialize_all_variables())
         n_epochs=1
         for epoch in range(n_epochs):
             epoch_loss=0
             i=0
             while i < train_x.shape[1]:
                   start = i
                   end = i+batch_size
                   batch_x = train_x[:,start:end]
                   batch_y = train_y[:,start:end]
                   batch_weight= weight[:,start:end]
              
                   feed_dict = {enc_inp[t]: batch_x[t] for t in range(seq_length)}
                   feed_dict.update({labels[t]: batch_y[t] for t in range(seq_length)})
                   feed_dict.update({weights[t]: batch_weight[t] for t in range(seq_length)})
              
                   _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
                   epoch_loss += loss_t
                   i+=batch_size

         print('Epoch', epoch+1, 'completed out of',n_epochs,'loss:',epoch_loss) 

                
     
         X_test=train_x[:,2:3]
         Y_test=train_y[:,2:3]

         feed_dict_test = {enc_inp[t]: X_test[t] for t in range(seq_length)}
         feed_dict_test.update({labels[t]: Y_test[t] for t in range(seq_length)})

         dec_outputs_test_batch = sess.run(decode_outputs_test, feed_dict_test)     
     
     

         t=X_test.flatten()
         tok=[rev_vocab.get(i) for i in t]
         print(len(tok))
         n_x_token=[]
         for i in tok:
             if(i!= None):
               n_x_token.append(i)

         print('n_x_token', n_x_token)
         print('ntl',len(n_x_token))
         h=len(n_x_token)
         c = ''.join(n_x_token);
         print('Input_tr',c)





         o=Y_test.flatten()
         tok=[rev_vocab.get(i) for i in o]
         print(len(tok))
         n_y_token=[]
         for i in tok:
             if(i!= None):
                n_y_token.append(i)
 
         c = ''.join(n_y_token);
         print('Target_tr',c)
     



         p=[logits_t.argmax(axis=1) for logits_t in dec_outputs_test_batch]
         m=np.array(p)
         print(len(m))
         n=m.flatten()
         o=n[0:h]
         print('lofo',len(o))


         token=[rev_vocab.get(i) for i in o]
     
         n_o_token=[]
         for i in token:
             if(i!= None):
                n_o_token.append(i)
     
         print('ntl',len(n_o_token))
         c = ''.join(n_o_token);
         print('Output_tr',c)
     
         save_path=saver.save(sess, "/home/ac/pp/mfinal.ckpt")
         print("Model saved in file: %s" % save_path)

     









     


def decode():
    with tf.Session() as sess:

         ckpt = tf.train.get_checkpoint_state("/home/ac/pp")
         if ckpt and "/home/ac/pp/mfinal.ckpt":
            print("model restored")
            saver.restore(sess,"/home/ac/pp/mfinal.ckpt")
                 
         else:
           print('created model with fresh parameters')
           sess.run(tf.initialize_all_variables())

         X_test=test_x
         Y_test=test_y

         feed_dict = {enc_inp[t]: X_test[t] for t in range(seq_length)}
         feed_dict.update({labels[t]: Y_test[t] for t in range(seq_length)})

         dec_outputs_batch = sess.run(dec_outputs, feed_dict)
     
     

         t=X_test.flatten()
         tok=[rev_vocab.get(i) for i in t]
     
         n_x_token=[]
         for i in tok:
             if(i!= None):
                n_x_token.append(i)
     
         h=len(n_x_token)
         c = ''.join(n_x_token);
         print('Input_dev',c)
 





         o=Y_test.flatten()
         tok=[rev_vocab.get(i) for i in o]
     
         n_y_token=[]
         for i in tok:
             if(i!= None):
                n_y_token.append(i)
 
         c = ''.join(n_y_token);
         print('Target_dev',c)



         p=[logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]
         m=np.array(p)
         n=m.flatten()
         o=n[0:h]

         token=[rev_vocab.get(i) for i in o]
     
         n_o_token=[]
         for i in token:
             if(i!= None):
                n_o_token.append(i)
 
         c = ''.join(n_o_token);
         print('Output_dev',c)
     


def main(_):
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()     

     

     
         

         
         
         

     

     



        
     
        