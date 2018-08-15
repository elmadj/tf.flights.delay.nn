from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import keras as k

from keras.layers import Dense, Dropout, Activation, Concatenate
from keras.optimizers import SGD

import os
import argparse
import shutil
import math
import sys
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tensorflow.python.ops import variables
import logging
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Merge, Flatten, Input, concatenate
from keras.regularizers import l1_l2
from keras.models import Model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=800, help='Number of training epochs.')

parser.add_argument(
    '--learning_rate', type=float, default=[0.01], nargs='+',
    help='The learning_rate.')

parser.add_argument(
    '--decay', type=float, default=0.000001,
    help='The decay.')

parser.add_argument(
    '--L1', type=float, default=0.0,
    help='The l1 regularization coeff.')

parser.add_argument(
    '--momentum', type=float, default=0.0,
    help='The momentum.')

parser.add_argument(
    '--L2', type=float, default=0.0,
    help='The l2 regularization coeff.')

parser.add_argument(
    '--batch_size', type=int, default=300, help='Number of examples per batch.')

parser.add_argument(
    '--all_data', type=str, default='',
    help='Path to the test data.')

parser.add_argument('--where', type=str, default='gpu', help='cpu of gpu')

parser.add_argument(
    '--airport', type=int, default=0,
    help='airport number.')

parser.add_argument(
    '--root_dir', type=str, default='./', help='root directory')

def setup_data(flightsfile, airport):
    flights = pd.read_csv(flightsfile)
    flights = flights[flights['ORIGIN_AIRPORT'] == airport]
    
    flights.reset_index(inplace=True)

    grouped = flights.groupby('ARRIVAL_DELAY_LABEL')
    #create a balanced train, validation and test set.
    sampled_indices_train = []
    sampled_indices_test = []
    sampled_indices_validation = []

    sampling_size =  grouped.size().max()

    for i in range(0, 7):
        print('Sampling class', i)
        sampled_indices_group = np.array(np.random.choice(grouped.groups[i], sampling_size))
        sampled_indices_train_group, sampled_indices_test_group = train_test_split(sampled_indices_group , test_size=0.20, random_state=42)
        sampled_indices_train_group, sampled_indices_validation_group = train_test_split(sampled_indices_train_group , test_size=0.20, random_state=42)

        sampled_indices_train = np.concatenate([sampled_indices_train, sampled_indices_train_group])
        sampled_indices_test = np.concatenate([sampled_indices_test, sampled_indices_test_group])
        sampled_indices_validation = np.concatenate([sampled_indices_validation, sampled_indices_validation_group])

    trainSet = flights.iloc[sampled_indices_train,:]
    testSet = flights.iloc[sampled_indices_test,:]
    validationSet = flights.iloc[sampled_indices_validation,:]

    return flights, trainSet, validationSet, testSet

# In[95]:
def get_embedding_dimension(vocabulary_size):
    return math.ceil(vocabulary_size**(1/float(4)))

def BuildFeedForwardNNClassifier(NonCategoricalInputs, CatInputs, Outputs, denseLayersFactor, nbHiddenLayers, activation, optimizer, L1, L2):
    Inputs = []
    sum_of_all_dimensions = 0

    i__s = (1, 1,)
    NonCatNs = []
    for column in NonCategoricalInputs:
        inputN = Input(shape=i__s, dtype=column.dtype, name='input_'+column.name)
        print(inputN)
        NonCatNs = [inputN]+NonCatNs 
        Inputs = [inputN]+Inputs

        sum_of_all_dimensions += 1 

    i_s = (1,)
    CatNs = []
    for column in CatInputs:
        number_of_categories = column.unique().size
        inputN = Input(shape=i_s, dtype='int32', name='input_'+column.name)
        embedding_dimension = get_embedding_dimension(number_of_categories)
        encoderN = Embedding(output_dim=embedding_dimension, input_dim=number_of_categories, input_length=1, name='embedding_'+column.name)(inputN)
        CatNs = [encoderN]+CatNs
        print(encoderN)
        Inputs = [inputN]+Inputs 
        
        sum_of_all_dimensions += embedding_dimension


    all_inputs = NonCatNs+CatNs
    I = concatenate(all_inputs)        
    dense_layer_dimension = math.ceil(denseLayersFactor * sum_of_all_dimensions)
    x = I
    for i in range(0, nbHiddenLayers):
        x = Dense(dense_layer_dimension, activation='relu', init='glorot_normal', activity_regularizer=l1_l2(L1, L2), name='hidden_'+str(i))(x)
        print('Building layer',i)

    number_of_output_classes = Outputs.unique().size
    main_output = Dense(number_of_output_classes, activation='softmax', name='main_output')(x)
    model = Model(inputs=Inputs, outputs=[main_output])

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



# In[145]:


# weightsVect = class_weight.compute_class_weight('balanced', [0,1,2,3,4,5,6], trainSet['ARRIVAL_DELAY_LABEL'])
# weightsVect


# # In[146]:


# weights = np.zeros(len(y_train))
# i=0
# for x in np.nditer(y_train):
#     weights[i] = weightsVect[x]
#     i+=1



if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    if FLAGS.where == 'gpu':
        num_GPU = 1
        num_CPU = 2
    if FLAGS.where == 'cpu':
        num_CPU = 2
        num_GPU = 0

    config = tf.ConfigProto(device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    k.backend.set_session(session)

    flights, trainSet, validationSet, testSet = setup_data(FLAGS.all_data, FLAGS.airport)

    size = trainSet.shape[0]
    val_size = validationSet.shape[0]
    input_train_data =  [trainSet['DESTINATION_AIRPORT'],
                        trainSet['TAIL_NUMBER'],
                        trainSet['FLIGHT_NUMBER'],
                        trainSet['AIRLINE'],
                        trainSet['DAY_OF_WEEK'],
                        trainSet['DAY'],
                        trainSet['MONTH'],
                        trainSet['SCHEDULED_ARRIVAL'].astype('float32').reshape((size, 1, 1)), 
                        trainSet['SCHEDULED_DEPARTURE'].astype('float32').reshape((size, 1, 1)), 
                        trainSet['DISTANCE'].astype('float32').reshape((size, 1, 1))]

    input_val_train_data =  [validationSet['DESTINATION_AIRPORT'],
                        validationSet['TAIL_NUMBER'],
                        validationSet['FLIGHT_NUMBER'],
                        validationSet['AIRLINE'],
                        validationSet['DAY_OF_WEEK'],
                        validationSet['DAY'],
                        validationSet['MONTH'],
                        validationSet['SCHEDULED_ARRIVAL'].astype('float32').reshape((val_size, 1, 1)), 
                        validationSet['SCHEDULED_DEPARTURE'].astype('float32').reshape((val_size, 1, 1)), 
                        validationSet['DISTANCE'].astype('float32').reshape((val_size, 1, 1))]
                        
    y_train = trainSet['ARRIVAL_DELAY_LABEL'].reshape((size, 1, 1))
    y_validation = validationSet['ARRIVAL_DELAY_LABEL'].reshape((val_size, 1, 1))

    for lr in FLAGS.learning_rate:
        print('Fitting model with learning rate = ', lr)
        sgd = SGD(lr=lr,
                decay=FLAGS.decay,
                momentum=FLAGS.momentum,
                nesterov=True)

        model = BuildFeedForwardNNClassifier([
                                    flights['DISTANCE'].astype('float32'), 
                                    flights['SCHEDULED_DEPARTURE'].astype('float32'), 
                                    flights['SCHEDULED_ARRIVAL'].astype('float32')], 
                                    [
                                    flights['MONTH'],
                                    flights['DAY'],
                                    flights['DAY_OF_WEEK'],
                                    flights['AIRLINE'],
                                    flights['FLIGHT_NUMBER'],
                                    flights['TAIL_NUMBER'],
                                    flights['DESTINATION_AIRPORT']], 
                                    flights['ARRIVAL_DELAY_LABEL'], 1.2, 1, 'sigmoid',sgd,
                                    FLAGS.L1,
                                    FLAGS.L2)
        print(model.summary())       
        
        model_directory = FLAGS.model_dir+'_lr'+str(lr)
        tbCallBack = TensorBoard(log_dir=model_directory,
                            histogram_freq=0,
                            write_graph=True,
                            write_images=False)

        model.fit(x=input_train_data, y=y_train, callbacks=[tbCallBack], batch_size=FLAGS.batch_size,
                epochs=FLAGS.train_epochs, validation_data=(input_val_train_data, y_validation), shuffle=True)
                #/, sample_weight=weights
