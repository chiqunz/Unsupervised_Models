import numpy as np
import tensorflow as tf


def seed():
    np.random.seed(0)
    tf.set_random_seed(0)


def xavier_init(num_row, num_col, const=1):
    low = -const * np.sqrt(6.0 / (num_row + num_col))
    high = const * np.sqrt(6.0 / (num_row + num_col))
    return tf.random_uniform(
        (num_row, num_col), minval=low, maxval=high, dytpe=tf.float32
    )  


class VAE:

    def __init__(self, num_hidden=16, input_shape=28*28, batch_size=36, learning_rate=1e-4):
        self.num_hidden = 16
        self.input_shape = input_shape
        self.batch_size = batch_size

        # Place holder for some key variables
        self.activation = None
        self.latent_mean = None
        self.latent_sigma = None
        self.inputs = None
        self.outputs = None
        self.z = None
        self.optimizer = None
        self.loss = None
        # Build the network
        self._build(learning_rate=learning_rate)
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def fit(self, X, epochs=100, shuffle=True):
        cost_hist = []
        num_batches = len(X) // self.batch_size
        for i in range(epochs):
            cost_epoch = 0
            if shuffle:
                np.random.shuffle(X)
            for j in range(num_batches):
                mini_X = X[j*self.batch_size:(j+1)*self.batch_size]
                cost_poch += self.one_pass(mini_X)
            cost_hist.append(cost_epoch / num_batches)
            if i % 50 == 0:
                self.saver.save(self.sess, "model.ckpt")
                print(cost_epoch / num_batches)
        self.saver.save(self.sess, "model.ckpt")
        return cost_hist

    def fit_generator(self, generator, num_samples, epochs=100):
        cost_hist = []
        num_batches = num_samples // self.batch_size
        for i in range(epochs):
            cost_epoch = 0
            for j in range(num_batches):
                mini_X, _ = generator(self.batch_size)
                cost_poch += self.one_pass(mini_X)
            cost_hist.append(cost_epoch / num_batches)
            if i % 50 == 0:
                self.saver.save(self.sess, "model.ckpt")
                print(cost_epoch / num_batches)
        self.saver.save(self.sess, "model.ckpt")
        return cost_hist

    def precit(self, X):
        return self.sess.run(self.outputs, feed_dict={self.inputs: X})

    def generate(self, z=None):
        if not z:
            z = np.random.normal(size=self.num_hidden)
        return self.sess.run(self.outputs, feed_dict={self.z: z})

    def encode(self, X):
        return self.sess.run(self.z, feed_dict={self.inputs: X})

    def one_pass(self, X):
        _, cost = self.sess.run((self.optimizer, self.loss), feed_dict={self.inputs: X})
        return cost

    def _build(self, learning_rate):
        self._build_network()
        self._create_loss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def _build_network(self, activation=tf.nn.relu):
        weights, bias = self._initialize_weights()
        self.activation = activation
        self.inputs = tf.placeholder(tf.float32, [None, self.input_shape])
        self.latent_mean, self.latent_sigma = self._build_encoder(weights['encoder'], bias['encoder'])
        eps = tf.random_normal((self.batch_size, self.num_hidden), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.latent_mean, tf.mul(self.latent_sigma, eps))
        self.outputs = self._build_decoder(weights['decoder'], bias['decoder'])

    def _create_loss(self):
        reconstruction_loss = - tf.reduce_sum(
            self.inputs * tf.log(self.outputs + 1e-5) + (1 - self.inputs) * tf.log(1 - self.outputs + 1e-5),
            axis=1
        )
        kl_loss = -0.5 * tf.reduce_sum(
            1 + tf.log(tf.square(self.latent_sigma))
            - tf.square(self.latent_mean)
            - tf.square(self.latent_sigma),
            axis=1
        )
        self.loss = tf.reduce_mean(reconstruction_loss, kl_loss)

    def _initialize_weights(self):
        weights = dict()
        weights['encoder'] = {
            'w1': tf.Variable(xavier_init(self.input_shape, 32)),
            'w2': tf.Variable(xavier_init(32, 16)),
            'latent_mean': tf.Variable(xavier_init(16, self.num_hidden)),
            'latent_sigma': tf.Variable(xavier_init(16, self.num_hidden))
        }
        weights['decoder'] = {
            'w1': tf.Variable(xavier_init(self.num_hidden, 16)),
            'w2': tf.Variable(xavier_init(16, 32)),
            'reconstruct': tf.Variable(xavier_init(32, self.input_shape))
        }
        bias = dict()
        bias['encoder'] = {
            'b1': tf.Variable(tf.zeros([32], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([16], dtype=tf.float32)),
            'latent_mean': tf.Variable(tf.zeros([self.num_hidden], dtype=tf.float32)),
            'latent_sigma': tf.Variable(tf.zeros([self.num_hidden], dtype=tf.float32)),
        }
        bias['decoder'] = {
            'b1': tf.Variable(tf.zeros([16], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([32], dtype=tf.float32)),
            'reconstruct': tf.Variable(tf.zeros([self.input_shape], dtype=tf.float32)),
        }
        return weights, bias

    def _build_encoder(self, weights, bias):
        l1 = self.activation(
            tf.add(tf.matmul(self.inputs, weights['w1']), bias['b1'])
        )
        l2 = self.activation(
            tf.add(tf.matmul(l1, weights['w2']), bias['b2'])
        )
        latent_mean = tf.nn.selu(
            tf.add(tf.matmul(l2, weights['latent_mean']), bias['latent_mean'])
        )
        latent_sigma = tf.nn.relu(
            tf.add(tf.matmul(l2, weights['latent_sigma']), bias['latent_sigma'])
        )
        latent_sigma = latent_sigma + 1e-5
        return latent_mean, latent_sigma

    def _build_decoder(self, weights, bias):
        l1 = self.activation(
            tf.add(tf.matmul(self.z, weights['w1']), bias['b1'])
        )
        l2 = self.activation(
            tf.add(tf.matmul(l1, weights['w2']), bias['b2'])
        )
        reconstruct = tf.nn.sigmoid(
            tf.add(tf.matmul(l2, weights['reconstruct']), bias['reconstruct'])
        )
        return reconstruct
