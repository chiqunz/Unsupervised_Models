import pickle
import logging
import numpy as np
from numpy import random as rng
import keras.backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import (Dense, Input, Activation, Reshape,
                          Conv1D, MaxPooling1D, Flatten, concatenate)
from keras.optimizers import Adam
from keras.utils import np_utils
import keras
import tensorflow as tf
from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder



def seeds():
    """Set same seed for every run"""
    K.clear_session()
    tf.set_random_seed(1)
    # pylint: disable=no-member
    rng.seed(1)


class ClusteringLayer(Layer):
    def __init__(self, output_dim, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.alpha = alpha
        self.initial_weights = weights
        self.centers = None
        super(ClusteringLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.centers = K.variable(self.initial_weights)
        self.built = True
        self.trainable_weights = [self.centers]
        super(ClusteringLayer, self).build(input_shape)


    def call(self, x):
        probq = 1.0/(1.0 + K.sum(K.square(K.expand_dims(x, 1) - self.weights), axis=2) / self.alpha)
        probq = probq**((self.alpha+1.0)/2.0)
        probq = K.transpose(K.transpose(probq)/K.sum(probq, axis=1))
        return probq


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class DeepEmbeddingClusteringSemi(object):
    def __init__(self,
                 input_dim=128,
                 alpha=1.0,
                 cluster_centers=None,
                 batch_size=256,
                 summary=False):

        self.input_dim = input_dim
        self.alpha = alpha
        self.cluster_centers = cluster_centers
        self.batch_size = batch_size
        self.summary = summary
        self.n_clusters = self.encoder = self.cnn = self.dec \
        = self.accuracy = self.y_pred = self.probp = self.probq = None

    def stoch_cnn(self):
        """build a CNN network"""
        seeds()
        act = Activation('selu')

        input_td = Input(shape=(self.input_dim,), name='input0')
        x_td = Reshape((128, 1))(input_td)
        for _ in range(4):
            x_td = Conv1D(16, kernel_size=4)(x_td)
            x_td = act(x_td)
            x_td = MaxPooling1D()(x_td)
        x_td = Flatten()(x_td)
        for _ in range(3):
            x_td = Dense(self.n_clusters)(x_td)
            x_td = act(x_td)
        p_td = Dense(self.n_clusters, activation='softmax')(x_td)

        input_fd = Input(shape=(self.input_dim,), name='input1')
        x_fd = Reshape((128, 1))(input_fd)
        for _ in range(4):
            x_fd = Conv1D(16, kernel_size=4)(x_fd)
            x_fd = act(x_fd)
            x_fd = MaxPooling1D()(x_fd)
        x_fd = Flatten()(x_fd)
        for _ in range(3):
            x_fd = Dense(self.n_clusters)(x_fd)
            x_fd = act(x_fd)
        p_fd = Dense(self.n_clusters, activation='softmax')(x_fd)

        mrg = concatenate([p_td, p_fd], name='embedded')
        prob = Dense(self.n_clusters, activation='softmax')(mrg)
        model = Model(inputs=[input_td, input_fd], outputs=prob)
        opt = Adam()

        if self.summary:
            model.summary()
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


    def decs(self):
        act = Activation('selu')
        input_td = Input(shape=(self.input_dim,), name='input0')
        x_td = Reshape((128, 1))(input_td)
        for _ in range(4):
            x_td = Conv1D(16, kernel_size=4)(x_td)
            x_td = act(x_td)
            x_td = MaxPooling1D()(x_td)
        x_td = Flatten()(x_td)
        for _ in range(3):
            x_td = Dense(self.n_clusters)(x_td)
            x_td = act(x_td)
        p_td = Dense(self.n_clusters, activation='softmax')(x_td)

        input_fd = Input(shape=(self.input_dim,), name='input1')
        x_fd = Reshape((128, 1))(input_fd)
        for _ in range(4):
            x_fd = Conv1D(16, kernel_size=4)(x_fd)
            x_fd = act(x_fd)
            x_fd = MaxPooling1D()(x_fd)
        x_fd = Flatten()(x_fd)
        for _ in range(3):
            x_fd = Dense(self.n_clusters)(x_fd)
            x_fd = act(x_fd)
        p_fd = Dense(self.n_clusters, activation='softmax')(x_fd)

        mrg = concatenate([p_td, p_fd], name='embedded')
        y = ClusteringLayer(self.n_clusters, weights=self.cluster_centers)(mrg)

        model = Model(inputs=[input_td, input_fd], outputs=y)
        opt = Adam()
        model.compile(optimizer=opt, loss='kullback_leibler_divergence',
                      metrics=['kullback_leibler_divergence',])
        return model

    def supervised_pretrain(self, data_labelled, labels, epochs=10000, save_cnn=False):
        #Pretrain the cnn on labelled data
        early = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=50,
            verbose=0,
            mode='auto'
        )
        tmpfilename = f'cnn_pretrain.hdf5'
        best = keras.callbacks.ModelCheckpoint(
            filepath=tmpfilename,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            period=1
        )

        self.encoder = LabelEncoder()
        self.encoder.fit(labels)
        encoded_labels = self.encoder.transform(labels)
        labels_one_hot = np_utils.to_categorical(encoded_labels)
        self.n_clusters = labels_one_hot.shape[1]
        self.input_dim = data_labelled[0].shape[1]


        self.cnn = self.stoch_cnn()
        self.cnn.fit(x=data_labelled, y=labels_one_hot, validation_split=0.1,
                     batch_size=self.batch_size, epochs=epochs, callbacks=[early, best],
                     verbose=2)
        self.cnn.load_weights(tmpfilename)
        if save_cnn:
            self.cnn.save_weights('CNN_weights_semi.h5')

        get_embedded = K.function([self.cnn.get_layer('input0').input,
                                   self.cnn.get_layer('input1').input],
                                  [self.cnn.get_layer('embedded').output])

        cluster_centers = []
        pred_labels = self.cnn.predict(data_labelled)
        pred_labels = pred_labels.argmax(axis=1)
        embedded_features = get_embedded(data_labelled)[0]
        for label in np.arange(self.n_clusters):
            center = embedded_features[pred_labels == label]
            if center.shape[0] == 0:
                # pylint: disable=no-member
                center = rng.rand(1, embedded_features.shape[1])
            cluster_centers.append(center.mean(axis=0))
        self.cluster_centers = np.vstack(cluster_centers)

        self.dec = self.decs()
        dec_weights = self.dec.get_weights()
        dec_weights[:-1] = self.cnn.get_weights()[:len(dec_weights) - 1]
        self.dec.set_weights(dec_weights)
        return

    def cluster(self, data_whole, y=None, tol=0.001, iter_max=1000,
                save_interval=10, epochs=10000):

        train = True
        iteration = 0
        self.accuracy = []

        get_embedded = K.function([self.dec.get_layer('input0').input,
                                   self.dec.get_layer('input1').input],
                                  [self.dec.get_layer('embedded').output])

        early = keras.callbacks.EarlyStopping(
            monitor='kullback_leibler_divergence',
            min_delta=0,
            patience=50,
            verbose=0,
            mode='auto'
            )
        pred_labels = self.cnn.predict(data_whole)
        self.y_pred = pred_labels.argmax(axis=1)

        while train:
            print('iteration: ' + str(iteration))
            tmpfilename = f'cluster_{iteration}.hdf5'
            best = keras.callbacks.ModelCheckpoint(
                filepath=tmpfilename,
                monitor='kullback_leibler_divergence',
                verbose=0,
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                period=1
                )
            if iteration > iter_max:
                print('Reach maximum iteration number. Exit.')
                return self.y_pred

            self.probq = self.dec.predict(data_whole, verbose=0)
            self.probp = p_cal(self.probq)

            y_pred = self.probq.argmax(axis=1)
            delta_label = ((y_pred != self.y_pred).sum() / y_pred.shape[0])
            if y is not None:
                y_encoded = self.encoder.transform(y)
                acc = cluster_acc(y_encoded, y_pred)
                self.accuracy.append(acc)
                print('Iteration '+str(iteration)+', Accuracy '+str(np.round(acc, 5)))
                print(str(np.round(delta_label*100, 5)) + '%' + ' change in label assignment')
            else:
                print(str(np.round(delta_label*100, 5)) + '%' + ' change in label assignment')

            if delta_label < tol and iteration != 0:
                print('Achieve tolerance. Exit.')
                train = False
                continue
            else:
                self.y_pred = y_pred
            self.cluster_centers = self.dec.layers[-1].get_weights()[0]

            self.dec.fit(data_whole, self.probp, batch_size=self.batch_size, epochs=epochs,
                         callbacks=[early, best], verbose=0)

            # save intermediate
            if iteration % save_interval == 0:
                print('Saving intermediate')
                dummyz = np.vstack(get_embedded(data_whole))
                dummyz = np.vstack([dummyz, self.cluster_centers])
                pca = decomposition.PCA(2)
                z_2d = pca.fit_transform(dummyz)
                point_2d = z_2d[:self.cluster_centers.shape[0]]
                clust_2d = z_2d[-self.cluster_centers.shape[0]:]
                # save states for visualization
                pickle.dump({'point_2d': point_2d, 'clust_2d': clust_2d,
                             'probq': self.probq, 'probp': self.probp},
                            open('c'+str(iteration)+'.pkl', 'wb'))
                # save dec model checkpoints
                self.dec.save('dec_model_semi_'+str(iteration)+'.h5')
            iteration += 1

        return self.y_pred


    def predict(self, x_test, y_test=None):
        """predict labels from saved model"""
        try:
            prob = self.dec.predict(x_test)
            y_pred = prob.argmax(axis=1)
            if y_test is not None:
                y_encoded = self.encoder.transform(y_test)
                accuracy = cluster_acc(y_encoded, y_pred)
                return y_pred, accuracy
        except AttributeError:
            logging.warning('No model is available.')
        return y_pred


def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    y_diff = np.abs(y_true - y_pred)
    y_diff[y_diff > 1] = 1
    return y_diff.sum()/y_true.shape[0]


def p_cal(probq):
    weight = probq**2 / probq.sum(axis=0)
    return weight / weight.sum(1)[:, None]
