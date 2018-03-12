# tf.flights.delay.nn
Create and train a neural net to classify flights

# This script tries to train a neural net to classify the delayed/canceled/diverted status of a flight based
# on a set of input features which represent the characteristics of a flight as described in the US department of transportation flight database
# The script can run in two modes : A prepare mode which pre-processes the data and splits it into train, validation and test set.
# A train mode which creates and trains a neural net model which can be monitored with tensorboard.
# In order to run this script, the root directory must be modified to match the directory where the data is placed.
# Data can be downloaded at https://drive.google.com/drive/folders/19oVuHTETul5BCPxUWU0aWQ_MTN3MAdmC?usp=sharing 
# It can be started for example with the command "python tf.flights.delay.nn.py --model_dir=Momentum_0.9_bs200_lr0.001 --learning_rate=0.001 --momentum=0.9 --batch_size=200"
