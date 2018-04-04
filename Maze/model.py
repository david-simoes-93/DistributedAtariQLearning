import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.models import Model
from keras.utils import to_categorical

def build_network(num_actions, agent_history_length, resized_width, resized_height, name_scope):
    with tf.device("/cpu:0"):
        with tf.name_scope(name_scope):
            state = tf.placeholder(tf.float32, [None, 40], name="state")
            inputs = Input(shape=(40,))
            model = Dense(60, activation='relu')(inputs)
            #print(model)
            q_values = Dense(num_actions)(model)

            #UserWarning: Update your `Model` call to the Keras 2 API:
            # `Model(outputs=Tensor("de..., inputs=Tensor("in..
            m = Model(inputs=inputs, outputs=q_values)
        
    return state, m