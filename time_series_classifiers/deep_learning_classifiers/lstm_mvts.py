import os
import os
import sys
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

import numpy as np
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, multiply, concatenate, Activation, Masking, Reshape
from keras.models import Model
from keras import backend as K

from time_series_classifiers.deep_learning_classifiers.dl_utils.keras_utils import train_model, evaluate_model
from time_series_classifiers.deep_learning_classifiers.dl_utils.layer_utils import AttentionLSTM

TRAINABLE = True


def generate_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

    # x = Permute((2, 1))(ip)
    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model


def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


def read_dataset(data_path):
    x_train = np.load(os.path.join(data_path, "{}.npy".format("TRAIN_X")))
    y_train = np.load(os.path.join(data_path, "{}.npy".format("TRAIN_Y")))
    x_test = np.load(os.path.join(data_path, "{}.npy".format("TEST_X")))
    y_test = np.load(os.path.join(data_path, "{}.npy".format("TEST_Y")))
    print("Data shape ", x_train.shape, x_test.shape)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    base_path = sys.argv[1]
    exercise = sys.argv[2]
    data_name = "MulticlassSplit"
    input_data_path = os.path.join(base_path, exercise, data_name)
    x_train, y_train, x_test, y_test = read_dataset(input_data_path)
    y_train = np.array([i-1 for i in y_train])
    y_test = np.array([i-1 for i in y_test])

    MAX_TIMESTEPS = x_train.shape[1]  # max time stamps
    MAX_NB_VARIABLES = x_train.shape[2]  # number of attributes
    NB_CLASS = len(np.unique(y_train))  # number of classes

    x_train = x_train.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)
    # A 0, Arch 1, N 2, R 3

    model = generate_model()
    train_model(model, exercise, x_train, y_train, x_test, y_test, epochs=600)
    evaluate_model(model, exercise, x_test, y_test)
    K.clear_session()

"""
 - 9s - loss: 2.0659e-04 - accuracy: 1.0000 - val_loss: 2.9644 - val_accuracy: 0.6054

PROJECT_PATH=/home/ashish/Research/Codes/human_pose_estimation/
DATA_PATH=/home/ashish/Results/Datasets/HPE
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH
exercise=MP_Video
cd $PROJECT_PATH/deep_learning_classifiers
CUDA_VISIBLE_DEVICES="" python lstm_mvts.py $DATA_PATH/TrainTestData_70_30 $exercise

rsync -avzhe 'ssh -J 19205522@resit-ssh.ucd.ie' --exclude '.git' --exclude '.idea' --exclude '.DS_Store' /Users/ashishsingh/Results/Datasets/HPE/TrainTestData_70_30  ashish@theengine.ucd.ie:/home/ashish/Results/Datasets/HPE/

alias code="cd /home/people/19205522/Research/Codes/human_pose_estimation/"
alias data="cd /home/people/19205522/scratch/Results/Datasets/HPE"

tmux new -s session_name
https://linuxize.com/post/getting-started-with-tmux/
"""