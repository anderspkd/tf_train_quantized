#!/usr/bin/env python

# Trains an mnist model (supplied as a commandline argument). Use mnist_freeze

import tensorflow as tf
from tensorflow import keras
from training_utils import parse_args

# may cause script to exit
args = parse_args()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# reshape and normalize
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],  1) / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) / 255.
# one hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# setup graph and set the session that keras will use.
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)
keras.backend.set_session(train_sess)

with train_graph.as_default():
    # build model
    model = args['model_fn']()

    # create a quantized training graph. This inserts fake quantization nodes
    # that emulate quantization when it's actually used in the inference phase.
    tf.contrib.quantize.create_training_graph(
        train_graph
    )

    # run, compile and train the model
    train_sess.run(tf.global_variables_initializer())
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=args['epochs'], batch_size=256)

    # save stuff so we can resume training if need be. Also needed when we want
    # to extract the quantized model (the tflite file).
    saver = tf.train.Saver()
    saver.save(train_sess, args['checkpoint_dir'])

    loss, accuracy = model.evaluate(x_test, y_test)
    print('\nEvaluation results:')
    print('test loss', loss)
    print('test accuracy', accuracy)
