#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import datetime
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.ops import variables
import logging

# This script tries to train a neural net to classify the delayed/canceled/diverted status of a flight based 
# on a set of input features which represent the characteristics of a flight as described in the US department of transportation flight database 
# The script can run in two modes : A prepare mode which preprocesses the data and splits it into train, validation and test set.
# A train mode which creates and trains a neural net model which can be monitored with tensorboard. 

#constants
root_dir = "C:\\Users\\aoudmadj\\Documents\\data science lab\\flight-delays.2"
raw_data_file = root_dir + "\\flights.csv"

training_data_file = root_dir + "\\flightsTrain.csv"
test_data_file = root_dir + "\\flightsTest.csv"
validation_data_file = root_dir + "\\flightsValidation.csv"





#utility function to add a label for DIVERTED and CANCELED flights
def transform_row(x):
   if x['DIVERTED'] == 1:
      x['ARRIVAL_DELAY_LABEL'] = 5
   if x['CANCELLED'] == 1:
      x['ARRIVAL_DELAY_LABEL'] = 6

# prepare and split the data files into train, validation and test sets.
def prepare_data():
    flights = pd.read_csv(raw_data_file)
#create a categorical output wi th 7 classes :
# class 0 : Flight arrival is earlier than expected
# class 2 : Flight arrival is 0 to 15 minutes late
# class 3 : Flight arrival is 15 to 30 minutes late
# class 4 : Flight arrival is 30 to 60 minutes late
# class 5 : Flight is DIVERTED
# class 6 : Flight is canceled

    bins = [-np.inf, 0, 15, 30, 60, np.inf]
    group_names = [0, 1, 2, 3, 4]
    # sort for performance
    flights.sort_values('ARRIVAL_DELAY', axis=0, ascending=False)
    flights['ARRIVAL_DELAY_LABEL'] = pd.cut(flights['ARRIVAL_DELAY'], bins, labels=group_names)
    flights.apply(lambda x: transform_row(x), axis=1,raw=True)

    grouped = flights.groupby('ARRIVAL_DELAY_LABEL')

# We have a lot of data so we take the min number of samples for all classes to
# avoid class imbalance
    number_samples = grouped['ARRIVAL_DELAY'].count().min()
    print('number of samples=', number_samples)

# create a balanced train, validation and test set.
    sampled_indices_train = []
    sampled_indices_test = []
    sampled_indices_validation = []

    for i in range(0, 6):
        print('Sampling class %d', i)
        sampled_indices_group = np.array(np.random.choice(grouped.groups[i], number_samples))
        sampled_indices_train_group, sampled_indices_test_group = train_test_split(sampled_indices_group , test_size=0.20, random_state=42)
        sampled_indices_train_group, sampled_indices_validation_group = train_test_split(sampled_indices_train_group , test_size=0.20, random_state=42)

        sampled_indices_train = np.concatenate([sampled_indices_train, sampled_indices_train_group])
        sampled_indices_test = np.concatenate([sampled_indices_test, sampled_indices_test_group])
        sampled_indices_validation = np.concatenate([sampled_indices_validation, sampled_indices_validation_group])

    trainSet = flights.iloc[sampled_indices_train,:]
    testSet = flights.iloc[sampled_indices_test,:]
    validationSet = flights.iloc[sampled_indices_validation,:]

    #normalize time and distance columns in train, test and validation sets.
    #Use the same scaler.
    print('Scaling...')
    #SCHEDULED_DEPARTURE
    scaler = StandardScaler()
    trainSet.SCHEDULED_DEPARTURE.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))
    validationSet.SCHEDULED_DEPARTURE.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))

    trainSet.SCHEDULED_DEPARTURE = scaler.fit_transform(trainSet.SCHEDULED_DEPARTURE.reshape(-1, 1))
    validationSet.SCHEDULED_DEPARTURE = scaler.transform(validationSet.SCHEDULED_DEPARTURE.reshape(-1, 1))

    testSet.SCHEDULED_DEPARTURE.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))
    testSet.SCHEDULED_DEPARTURE = scaler.transform(testSet.SCHEDULED_DEPARTURE.reshape(-1, 1))

    #DEPARTURE_TIME
    scaler = StandardScaler()
    trainSet.DEPARTURE_TIME.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))
    validationSet.DEPARTURE_TIME.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))

    trainSet.DEPARTURE_TIME = scaler.fit_transform(trainSet.DEPARTURE_TIME.reshape(-1, 1))
    validationSet.DEPARTURE_TIME = scaler.transform(validationSet.DEPARTURE_TIME.reshape(-1, 1))

    testSet.DEPARTURE_TIME.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))
    testSet.DEPARTURE_TIME = scaler.transform(testSet.DEPARTURE_TIME.reshape(-1, 1))

    #SCHEDULED_TIME
    scaler = StandardScaler()
    trainSet.SCHEDULED_TIME.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))
    validationSet.SCHEDULED_TIME.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))

    trainSet.SCHEDULED_TIME = scaler.fit_transform(trainSet.SCHEDULED_TIME.reshape(-1, 1))
    validationSet.SCHEDULED_TIME = scaler.transform(validationSet.SCHEDULED_TIME.reshape(-1, 1))

    testSet.SCHEDULED_TIME.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))
    testSet.SCHEDULED_TIME = scaler.transform(testSet.SCHEDULED_TIME.reshape(-1, 1))

    #SCHEDULED_ARRIVAL
    scaler = StandardScaler()
    trainSet.SCHEDULED_ARRIVAL.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))
    validationSet.SCHEDULED_ARRIVAL.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))

    trainSet.SCHEDULED_ARRIVAL = scaler.fit_transform(trainSet.SCHEDULED_ARRIVAL.reshape(-1, 1))
    validationSet.SCHEDULED_ARRIVAL = scaler.transform(validationSet.SCHEDULED_ARRIVAL.reshape(-1, 1))

    testSet.SCHEDULED_ARRIVAL.apply(lambda x: (np.floor(x / 100) * 60 + x % 100))
    testSet.SCHEDULED_ARRIVAL = scaler.transform(testSet.SCHEDULED_ARRIVAL.reshape(-1, 1))


    #DISTANCE
    scaler = StandardScaler()
    trainSet.DISTANCE = scaler.fit_transform(trainSet.DISTANCE.reshape(-1, 1))
    validationSet.DISTANCE = scaler.transform(validationSet.DISTANCE.reshape(-1, 1))

    testSet.DISTANCE = scaler.transform(testSet.DISTANCE.reshape(-1, 1))
    print('Saving...')
    # save data
    trainSet.to_csv(training_data_file , index=False)
    testSet.to_csv(test_data_file , index=False)
    validationSet.to_csv(validation_data_file , index=False)


#column names
_CSV_COLUMNS = ['INDEX', 'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE',
   'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
   'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
   'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
   'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
   'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
   'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
   'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'ARRIVAL_DELAY_LABEL']


_CSV_COLUMN_DEFAULTS = [[0],
                        ['2015'],
                        ['1'],
                        ['1'],
                        ['1'],
                        [''],
                        [0],
                        [''],
                        [''],
                        [''],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0],
                        [0],
                        [''],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [1]]

#Argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default=root_dir + str(datetime.datetime.now().timestamp()),
    help='Base directory for the model.')

parser.add_argument('--model_type', type=str, default='deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument('--train_epochs', type=int, default=5000, help='Number of training epochs.')

parser.add_argument('--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')

parser.add_argument('--learning_rate', type=float, default=0.01,
    help='The learning_rate.')

parser.add_argument('--l1', type=float, default=0.0,
    help='The l1 regularization coeff.')

parser.add_argument('--momentum', type=float, default=0.0,
    help='The momentum.')

parser.add_argument('--l2', type=float, default=0.0,
    help='The l2 regularization coeff.')

parser.add_argument('--batch_size', type=int, default=300, help='Number of examples per batch.')

parser.add_argument('--train_data', type=str, default=training_data_file,
    help='Path to the training data.')

parser.add_argument('--validation_data', type=str, default=validation_data_file ,
    help='Path to the test data.')

parser.add_argument('--prepare', type=bool, default=False,
    help='Path to the test data.')

parser.add_argument('--restore', type=str, default='',
    help='Name of model to restore.')

#Build input features
def build_model_columns(flightsTrain):
  with tf.device('/CPU:0'):
    month = tf.feature_column.categorical_column_with_vocabulary_list('MONTH',
            ['1', '2', '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12'])

    day = tf.feature_column.categorical_column_with_vocabulary_list('DAY', ['1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                '10',
                '11',
                '12',
                '13',
                '14',
                '15',
                '16',
                '17',
                '18',
                '19',
                '20',
                '21',
                '22',
                '23',
                '24',
                '25',
                '26',
                '27',
                '28',
                '29',
                '30',
                '31'])

    day_of_week = tf.feature_column.categorical_column_with_vocabulary_list('DAY_OF_WEEK', ['1',
                    '2',
                    '3',
                    '4',
                    '5',
                    '6',
                    '7'])


    number_of_airlines = flightsTrain['AIRLINE'].unique().size
    number_of_airports = flightsTrain['ORIGIN_AIRPORT'].unique().size
    number_of_airplanes = flightsTrain['TAIL_NUMBER'].unique().size
    number_of_flights = flightsTrain['FLIGHT_NUMBER'].unique().size

    #create columns
    airline = tf.feature_column.categorical_column_with_vocabulary_list('AIRLINE', flightsTrain['AIRLINE'].unique())
    flight_number = tf.feature_column.categorical_column_with_hash_bucket('FLIGHT_NUMBER', flightsTrain['FLIGHT_NUMBER'].max(), dtype=tf.int64)
    tail_number = tf.feature_column.categorical_column_with_vocabulary_list('TAIL_NUMBER', flightsTrain['TAIL_NUMBER'].unique())
    origin_airport = tf.feature_column.categorical_column_with_vocabulary_list('ORIGIN_AIRPORT', flightsTrain['ORIGIN_AIRPORT'].unique())
    destination_airport = tf.feature_column.categorical_column_with_vocabulary_list('DESTINATION_AIRPORT', flightsTrain['DESTINATION_AIRPORT'].unique())
    scheduled_departure = tf.feature_column.numeric_column('SCHEDULED_DEPARTURE', dtype=tf.float32)
    departure_time = tf.feature_column.numeric_column('DEPARTURE_TIME', dtype=tf.float32)
    scheduled_time = tf.feature_column.numeric_column('SCHEDULED_TIME', dtype=tf.float32)
    scheduled_arrival = tf.feature_column.numeric_column('SCHEDULED_ARRIVAL', dtype=tf.float32)
    distance = tf.feature_column.numeric_column('DISTANCE', dtype=tf.float32)

    #cross_origin_airport_month_day =
    #tf.feature_column.crossed_column(keys=[month, day, origin_airport],
    #hash_bucket_size=120156)

    # Group columns into wide columns and deep columns.
    base_columns = [month,
            day,
            day_of_week,
            airline,
            flight_number ,
            tail_number,
            origin_airport,
            destination_airport,
            scheduled_departure ,
            departure_time ,
            scheduled_time ,
            scheduled_arrival,
            distance]

    crossed_columns = []

    wide_columns = base_columns + crossed_columns

    deep_columns = [# create embeddings for our input columns with many dimensions
            # take the fourth root of the input dimension as the embedding's
            # output dimension.
            tf.feature_column.embedding_column(month, math.ceil(12.0 ** (1 / float(4)))),
            tf.feature_column.embedding_column(day, math.ceil(31.0 ** (1 / float(4)))),
            tf.feature_column.embedding_column(day_of_week, math.ceil(7.0 ** (1 / float(4)))),
            tf.feature_column.embedding_column(airline, math.ceil(number_of_airlines ** (1 / float(4)))),
            tf.feature_column.embedding_column(flight_number , math.ceil(number_of_flights ** (1 / float(4)))),
			tf.feature_column.embedding_column(tail_number, math.ceil(number_of_airplanes ** (1 / float(4)))),
            tf.feature_column.embedding_column(origin_airport, math.ceil(number_of_airports ** (1 / float(4)))),
            tf.feature_column.embedding_column(destination_airport, math.ceil(number_of_airports ** (1 / float(4)))),
           # tf.feature_column.embedding_column(cross_origin_airport_month_day,
           # math.ceil(120156**(1/float(4)))),
            scheduled_departure ,
            departure_time ,
            scheduled_time ,
            scheduled_arrival,
            distance]

    #take the total size
    number_columns_in_input_layer = math.ceil((math.ceil(12.0 ** (1 / float(4))) + math.ceil(31.0 ** (1 / float(4))) + math.ceil(7.0 ** (1 / float(4))) + math.ceil(number_of_airlines ** (1 / float(4))) + math.ceil(number_of_flights ** (1 / float(4))) + math.ceil(number_of_airplanes ** (1 / float(4))) + math.ceil(number_of_airports ** (1 / float(4))) + math.ceil(number_of_airports ** (1 / float(4)))+5)/2)
    return wide_columns, deep_columns, number_columns_in_input_layer


def build_estimator(model_dir, model_type, TrainSet):
  #Build an estimator appropriate for the given model type.
  with tf.device('/CPU:0'):
      wide_columns, deep_columns, number_columns_in_input_layer = build_model_columns(trainSet)
      hidden_units = [math.ceil(number_columns_in_input_layer * 2 / 3), math.ceil(number_columns_in_input_layer / 3)]

      optimizer = tf.train.AdagradOptimizer(learning_rate = FLAGS.learning_rate, initial_accumulator_value=0.1)
      conf = tf.ConfigProto(log_device_placement=False)
      runConfig = tf.estimator.RunConfig(session_config=conf)
      if model_type == 'wide':
        return tf.estimator.LinearClassifier(model_dir=model_dir,
            feature_columns=wide_columns,
            n_classes=8,
            optimizer=tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum = FLAGS.momentum, use_nesterov=True),
            config=runConfig)
      elif model_type == 'deep':
        return tf.estimator.DNNClassifier(model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            n_classes=8,
            activation_fn=tf.nn.sigmoid,
            optimizer=tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum = FLAGS.momentum, use_nesterov=True))
      else:
        return tf.estimator.DNNLinearCombinedClassifier(model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units)

def input_fn(data_file, num_epochs, shuffle, batch_size, train_dataset_size):
  with tf.device('/CPU:0'):
      assert tf.gfile.Exists(data_file), ('%s not found. Please make sure you have either run data_download.py or '
          'set both arguments --train_data and --validation_data.' % data_file)

      def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('ARRIVAL_DELAY_LABEL')
        return features, labels

      # Extract lines from input files using the Dataset API.
      dataset = tf.data.TextLineDataset(data_file).skip(1)
      if shuffle:
        dataset = dataset.shuffle(buffer_size=train_dataset_size)
      dataset = dataset.map(parse_csv, num_parallel_calls=5)
      dataset = dataset.repeat(num_epochs)
      dataset = dataset.batch(batch_size)
      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
      return features, labels


def main(unused_argv):
    flightsTrain = pd.read_csv(train_data_file)
    #shutil.rmtree(root_dir + "\\" + FLAGS.model_dir, ignore_errors=True)
    model = build_estimator(root_dir + "\\" + FLAGS.model_dir, FLAGS.model_type, flightsTrain)
    #read the train dataset only to know its size
    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    with tf.device("/CPU:0"):
        with tf.Session() as sess:
            model_checkpoint_file_base = root_dir + '\\' + FLAGS.model_dir + '\\' + FLAGS.restore
            if FLAGS.restore != '':
                saver = tf.train.import_meta_graph(model_checkpoint_file_base + ".meta")
                saver.restore(sess, model_checkpoint_file_base)
                print("restored model ", model_checkpoint_file_base)
            sess.run(tf.global_variables_initializer())
            for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
                model.train(input_fn=lambda: input_fn(FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size, flightsTrain))

                results_validation = model.evaluate(input_fn=lambda: input_fn(FLAGS.validation_data, 1, False, FLAGS.batch_size), name='validation')

                # Display evaluation metrics
                print('validation Results at epoch', (n + 1) * FLAGS.epochs_per_eval)

                for key in sorted(results_validation):
                  print('%s: %s' % (key, results_validation[key]))


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS.batch_size)
    if FLAGS.prepare:
        prepare_data()
    else:
        with tf.Session():
            tf.logging.set_verbosity(tf.logging.INFO)
            summary_writer = tf.summary.FileWriter(root_dir + "\\" + FLAGS.model_dir, graph=tf.get_default_graph())
            tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




