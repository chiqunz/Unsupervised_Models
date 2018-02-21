# -*- coding: utf-8 -*-
"""A basic neural network based on tensorflow, taking comparison input"""

import tensorflow as tf
import numpy as np


class comparisonNN:

    """Define the model with fit and predict"""

    def __init__(self, epochs=10000, save_interval=500, confidence=1, hidden_layer_width=5):
        self.graph = tf.Graph()
        self.epochs = epochs
        self.save_interval = save_interval
        self.confidence = confidence
        self.learning_rate = 0.0001
        self.input_dim = None
        self.hidden_layer_width = hidden_layer_width
        self.output_dim = None

    def model_build(self):
        """Network structure.

        Parameters
        ----------
        X_labelled: labelled features
        Y_labelled: labelled targets
        X_comp1: the first one in each comparison pair
        X_comp2: the second one in each comparison pair
        Y_comp: comparison target, -1: X_comp1<X_comp2; 1: X_comp1>X_comp2

        Returns
        -----------
        TODO

        """

        #Graph and variables
        with self.graph.as_default():
            initializer = tf.keras.initializers.glorot_uniform()
            W1 = tf.Variable(initializer([self.input_dim, self.hidden_layer_width]))
            b1 = tf.Variable(tf.zeros([self.hidden_layer_width]))
            W2 = tf.Variable(initializer([self.hidden_layer_width, self.output_dim]))
            b2 = tf.Variable(tf.zeros([self.output_dim]))
            X_labelled = tf.placeholder(tf.float32, [None, self.input_dim], name='X_labelled')
            Y_labelled = tf.placeholder(tf.float32, name='Y_labelled')
            X_comp1 = tf.placeholder(tf.float32, [None, self.input_dim], name='X_comp1')
            X_comp2 = tf.placeholder(tf.float32, [None, self.input_dim], name='X_comp2')
            Y_comp = tf.placeholder(tf.float32, name='Y_comp')

            #Networks and models
            hidden_layer_labelled = tf.add(tf.matmul(X_labelled, W1), b1)
            hidden_layer_labelled = tf.nn.relu(hidden_layer_labelled)
            output_layer_labelled = tf.add(tf.matmul(hidden_layer_labelled, W2), b2, name='output')
            loss_labelled = tf.reduce_mean(tf.squared_difference(output_layer_labelled, Y_labelled),
                                           name='mse_labelled')
            hidden_layer_comp1 = tf.add(tf.matmul(X_comp1, W1), b1)
            hidden_layer_comp1 = tf.nn.relu(hidden_layer_comp1)
            output_layer_comp1 = tf.add(tf.matmul(hidden_layer_comp1, W2), b2)
            hidden_layer_comp2 = tf.add(tf.matmul(X_comp2, W1), b1)
            hidden_layer_comp2 = tf.nn.relu(hidden_layer_comp2)
            output_layer_comp2 = tf.add(tf.matmul(hidden_layer_comp2, W2), b2)
            output_layer_subtract = tf.subtract(output_layer_comp1, output_layer_comp2)
            output_comp = tf.cast(tf.greater(output_layer_subtract, 0), tf.float32)
            loss_comp = tf.reduce_mean(tf.squared_difference(output_comp, Y_comp), name='mse_comp')
            loss = tf.add(loss_labelled, self.confidence * loss_comp, name='loss')
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            self.init = tf.global_variables_initializer()
        return


    def fit(self, x_train_labelled, y_train_labelled, x_train_comp1, x_train_comp2, y_train_comp):
        self.input_dim = x_train_labelled.shape[1]
        self.output_dim = y_train_labelled.shape[1]
        self.model_build()
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init)

            # Fit all training data
            for epoch in range(self.epochs):
                sess.run(self.optimizer, feed_dict={'X_labelled:0': x_train_labelled, 
                                               'Y_labelled:0': y_train_labelled,
                                               'X_comp1:0': x_train_comp1,
                                               'X_comp2:0': x_train_comp2,
                                               'Y_comp:0': y_train_comp})

                # Display logs per epoch step
                if (epoch+1) % self.save_interval == 0:
                    c = sess.run('loss:0', feed_dict={'X_labelled:0': x_train_labelled, 
                                                  'Y_labelled:0': y_train_labelled,
                                                  'X_comp1:0': x_train_comp1,
                                                  'X_comp2:0': x_train_comp2,
                                                  'Y_comp:0': y_train_comp})
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            saver = tf.train.Saver()
            saver.save(sess, './model')
        return


    def predict(self, x_test):
        sess=tf.Session()
        saver = tf.train.import_meta_graph('./model' + '.meta')
        saver.restore(sess, 'model')
        y_pred = sess.run('output:0', feed_dict={'X_labelled:0': x_test})
        return y_pred







