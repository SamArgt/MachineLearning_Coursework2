import tensorflow as tf
from tensorflow.keras import layers


class NN():
  def __init__(self, n_neurons, lr, alpha, epochs=250, batch_size=64):
    self.alpha = alpha
    self.lr = lr
    self.epochs = epochs
    self.batch_size = batch_size

    self.mlp = tf.keras.Sequential()
    for n in n_neurons:
      self.mlp.add(layers.Dense(n, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(
                                    self.alpha),
                                bias_regularizer=tf.keras.regularizers.l2(self.alpha)))

    self.mlp.add(layers.Dense(1,
                              kernel_regularizer=tf.keras.regularizers.l2(
                                  self.alpha),
                              bias_regularizer=tf.keras.regularizers.l2(self.alpha)))

    self.mlp.compile(optimizer=tf.keras.optimizers.Adam(self.lr),
                     loss='mse',
                     metrics=[])

  def fit(self, X, y):

    self.mlp.fit(X, y, epochs=self.epochs,
                 batch_size=self.batch_size, verbose=0)

  def predict(self, X):
    return self.mlp.predict(X)
