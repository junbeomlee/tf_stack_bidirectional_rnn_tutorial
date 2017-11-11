
# coding: utf-8

# In[646]:


# from IPython import get_ipython
# get_ipython().magic('reset -sf') 

import tensorflow as tf
import numpy as np

tf.__version__



# In[647]:


# x->x를 예측

PAD = 0
START = 1
EOS = 2

num_class = 7
hidden_unit = 7
num_layer = 3
batch_size = 3


encoder_input = [
    [5,6,3,4],
    [6,5,3,0],
    [5,6,0,0]
]

decoder_input = [
    [START,5,6,3,4],
    [START,6,5,3,0],
    [START,5,6,0,0]
]

target = [
    [5,6,3,4,EOS],
    [6,5,3,EOS,0],
    [5,6,EOS,0,0]
]


# In[648]:


encoder_input


# In[649]:


decoder_input


# In[650]:


target


# In[651]:


def model_fn(mode, features, labels,params):
    if type(features) == dict:
        print(features)
        encoder_input = features["encoder_input"]
        decoder_input = features["decoder_input"]

    encoder_input_one_hot = tf.one_hot(encoder_input,depth=num_class,dtype=tf.float32)
    decoder_input_one_hot = tf.one_hot(decoder_input,depth=num_class,dtype=tf.float32)
    with tf.variable_scope('encoder'):
        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_unit,state_is_tuple=True)
        initial_state = cell.zero_state(batch_size,tf.float32)
        enc_outputs, enc_states = tf.nn.dynamic_rnn(
            cell,
            encoder_input_one_hot,
            initial_state=initial_state,
            dtype=tf.float32)
    with tf.variable_scope('decoder'):
        decoder_cell = tf.contrib.rnn.LSTMCell(num_units=hidden_unit,state_is_tuple=True)
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
            decoder_cell, decoder_input_one_hot,
            initial_state=enc_states,
            dtype=tf.float32)

    decoder_logits = tf.contrib.layers.fully_connected(decoder_outputs, num_class,activation_fn=None)
    decoder_prediction = tf.argmax(decoder_logits, 2)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=decoder_prediction)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels,depth=num_class,dtype=tf.float32),
            logits=decoder_logits,
        )
        
        loss = tf.reduce_mean(stepwise_cross_entropy,name="loss")
            
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer='Adam',
            summaries=['loss', 'learning_rate'],
            learning_rate=0.001,
            name="train_op")
        
        predictions = decoder_prediction
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            predictions=predictions,
            train_op=train_op)
   


# In[662]:


run_config = tf.contrib.learn.RunConfig(model_dir="./model_dir",save_summary_steps=500)


# In[663]:


# summary_hook = tf.train.SummarySaverHook(
#     10,
#     output_dir='./log',
#     summary_op=tf.summary.merge_all())


# In[664]:


estimator = tf.estimator.Estimator(config=run_config,model_fn=model_fn,model_dir="./model_dir",params=None,)


# In[665]:


my_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"encoder_input":np.array(encoder_input,dtype=np.int32),"decoder_input":np.array(decoder_input,dtype=np.int32)},
    y=np.array(target,dtype=np.int32),
    batch_size=batch_size,
    shuffle=False,
    num_epochs=None)




# In[667]:



estimator.train(input_fn = my_input_fn,hooks=None,steps=5000,max_steps=None,saving_listeners=None)


# In[ ]:


my_predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"encoder_input":np.array(encoder_input,dtype=np.int32),"decoder_input":np.array(decoder_input,dtype=np.int32)},
    y=np.array(decoder_input,dtype=np.int32),
    batch_size=batch_size,
    shuffle=False,
    num_epochs=1)


# In[ ]:


result = estimator.predict(input_fn=my_predict_input_fn)


# In[ ]:


for label,pred in zip(target,result):
    print(label,pred)


# In[ ]:


for i in tf.get_default_graph().get_operations():
    print (i.name)

