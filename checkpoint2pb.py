#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from models import models
import argparse

p = argparse.ArgumentParser(
    description='converts a frozen .pb model to .tflite'
)
p.add_argument(
    'model_name',
    help='name of model',
    type=str
)
p.add_argument(
    'checkpoints',
    help='checkpoints filename',
    type=str
)
p.add_argument(
    '-g',
    help='input/output names in grep friendly format',
    action='store_true'
)
args = p.parse_args()


def find_model_by_name(model_name):
    for name in models.keys():
        if name == model_name:
            if not args.g:
                print('found model:', name)
            return models[name]
    raise ValueError(f'model caled "{model_name}" not found')


graph = tf.Graph()
sess = tf.Session(graph=graph)
keras.backend.set_session(sess)

with graph.as_default():
    # restore model from checkpoint
    keras.backend.set_learning_phase(0)
    model = find_model_by_name(args.model_name)()
    tf.contrib.quantize.create_eval_graph(input_graph=graph)
    graph_def = graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoints)

    # freeze graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        [model.output.op.name]
    )

    model_filename = args.model_name + '.pb'

    if not args.g:
        print('------------------------')
        print('writing', model_filename)
        print('input_arrays:', model.input.op.name)
        print('output_arrays:', model.output.op.name)
    else:
        print('in/out:', model.input.op.name, model.output.op.name)

    with open(model_filename, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
