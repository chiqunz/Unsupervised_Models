import sys
import numpy as np
import keras.backend as K
from keras.initializers import RandomNormal
from keras.engine.topology import Layer, InputSpec
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from sklearn.preprocessing import normalize
import keras
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import manifold
if (sys.version[0] == 2):
    import cPickle as pickle
else:
    import pickle
import numpy as np

class ClusteringLayer(Layer):
    def __init__(self, output_dim, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.alpha = alpha
        self.initial_weights = weights
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = K.variable(self.initial_weights)
        self.built = True
        self.trainable_weights = [self.W]
        super(ClusteringLayer, self).build(input_shape)

    def call(self, x):
        q = 1.0/(1.0 + K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = K.transpose(K.transpose(q)/K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class DeepEmbeddingClustering(object):
    def __init__(self,
                 n_clusters,
                 input_dim=128,
                 alpha=1.0,
                 cluster_centers=None,
                 batch_size=256,
                 **kwargs):
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.alpha = alpha
        self.cluster_centers = cluster_centers
        self.batch_size = batch_size

    def autoencoders(self, dropout_fraction = 0.1):
        inputs = Input(shape=(self.input_dim,), name='input')

        x = Dense(200, activation='linear')(inputs)
        x = Dropout(dropout_fraction)(x)
        x = Dense(200, activation='selu')(x)
        x = Dropout(dropout_fraction)(x)
        x = Dense(200, activation='selu')(x)
        x = Dense(500, activation='selu')(x)

        f = Dense(32, activation='linear', name='embedded')(x)

        y = Dense(500, activation='selu')(f)
        y = Dense(200, activation='selu')(y)
        y = Dropout(dropout_fraction)(y)
        y = Dense(200, activation='selu')(y)
        y = Dropout(dropout_fraction)(y)
        y = Dense(200, activation='linear')(y)

        outputs = Dense(self.input_dim, activation='linear', name='output')(y)

        model = Model(inputs=inputs, outputs=outputs)
        opt=Adam()
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])

        return model


    def DECS(self, n_clusters, cluster_centers, dropout_fraction = 0.1):
        inputs = Input(shape=(self.input_dim,), name='input')

        x = Dense(200, activation='linear')(inputs)
        x = Dropout(dropout_fraction)(x)
        x = Dense(200, activation='selu')(x)
        x = Dropout(dropout_fraction)(x)
        x = Dense(200, activation='selu')(x)
        x = Dense(500, activation='selu')(x)

        f = Dense(32, activation='linear', name='embedded')(x)

        y = ClusteringLayer(n_clusters, weights=cluster_centers)(f)

        model = Model(inputs=inputs, outputs=y)
        opt=Adam()
        model.compile(optimizer=opt, loss='kullback_leibler_divergence', metrics=['kullback_leibler_divergence',])

        return model

    def initialize(self, X, epochs=10000, save_autoencoder=True):
        #Initalize the embedded space:
        early = keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=0,
                patience=50,
                verbose=0,
                mode='auto'
                )
        tmpfilename = f'autoencoder.hdf5'
        best = keras.callbacks.ModelCheckpoint(
                filepath=tmpfilename,
                monitor='loss',
                verbose=0,
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                period=1
                )
        self.autoencoder = self.autoencoders()

        self.autoencoder.fit(X, X, batch_size=self.batch_size,
                             epochs=epochs, callbacks=[early, best], verbose=2)
        self.autoencoder.load_weights(tmpfilename)
        if save_autoencoder:
            self.autoencoder.save_weights('autoencoder.h5')

        get_embedded = K.function([self.autoencoder.get_layer('input').input,
                               K.learning_phase()],
                               [self.autoencoder.get_layer('embedded').output])
        embedded = np.vstack(get_embedded([X, 0]))

        # initialize cluster centres using k-means
        print('Initializing cluster centres with k-means.')
        if self.cluster_centers is None:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            self.y_pred = kmeans.fit_predict(embedded)
            self.cluster_centers = kmeans.cluster_centers_


        self.DEC = self.DECS(self.n_clusters, self.cluster_centers)
        DEC_weights = self.DEC.get_weights()
        DEC_weights[:-1] = self.autoencoder.get_weights()[:len(DEC_weights)-1]
        self.DEC.set_weights(DEC_weights)
        return

    def cluster_acc(self, y_true, y_pred):
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max())+1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind])*1.0/y_pred.size, w

    def p_cal(self, q):
        weight = q**2 / q.sum(axis=0)
        return weight / weight.sum(1)[:, None]

    def cluster(self, X, y=None, tol=0.001, iter_max=1000, save_interval=10, **kwargs):

        train = True
        iteration = 0
        self.accuracy = []

        get_embedded = K.function([self.DEC.get_layer('input').input,
                                   K.learning_phase()],
                                  [self.DEC.get_layer('embedded').output])

        early = keras.callbacks.EarlyStopping(
                monitor='kullback_leibler_divergence',
                min_delta=0,
                patience=50,
                verbose=0,
                mode='auto'
                )

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

            self.q = self.DEC.predict(X, verbose=0)
            self.p = self.p_cal(self.q)

            y_pred = self.q.argmax(axis=1)
            delta_label = ((y_pred != self.y_pred).sum() / y_pred.shape[0])
            if y is not None:
                acc = self.cluster_acc(y, y_pred)[0]
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
            self.cluster_centers = self.DEC.layers[-1].get_weights()[0]

            self.DEC.fit(X, self.p, batch_size=self.batch_size, epochs=10000,
                         callbacks=[early, best], verbose=0)

            # save intermediate
            if iteration % save_interval == 0:
                z = np.vstack(get_embedded([X, 0]))
                z = np.vstack([z, self.cluster_centers])
                tsne = manifold.TSNE(n_components=2)
                z_2d = tsne.fit_transform(z)
                point_2d = z_2d[:self.cluster_centers.shape[0]]
                clust_2d = z_2d[-self.cluster_centers.shape[0]:]
                # save states for visualization
                pickle.dump({'z_2d': z_2d, 'clust_2d': clust_2d, 'q': self.q, 'p': self.p},
                            open('c'+str(iteration)+'.pkl', 'wb'))
                # save DEC model checkpoints
                self.DEC.save('DEC_model_'+str(iteration)+'.h5')

            iteration += 1
        return
