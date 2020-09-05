import tensorflow as tf

"""
Model class, taken from:
https://github.com/apoorvnandan/speech-recognition-primer
"""

class ASR(tf.keras.Model):
    """
    Class for defining the end-to-end ASR model.
    This model consists of a 1D convolutional layer followed by a bidirectional LSTM
    followed by a fully connected layer applied at each timestep.
    """
    def __init__(self, filters, kernel_size, conv_stride, conv_border, n_lstm_units, n_dense_units):
        super(ASR, self).__init__()
        self.conv_layer = tf.keras.layers.Conv1D(filters,
                                                 kernel_size,
                                                 strides=conv_stride,
                                                 padding=conv_border,
                                                 activation='relu')
        self.lstm_layer = tf.keras.layers.LSTM(n_lstm_units,
                                               return_sequences=True,
                                               activation='tanh')
        self.lstm_layer_back = tf.keras.layers.LSTM(n_lstm_units,
                                                    return_sequences=True,
                                                    go_backwards=True,
                                                    activation='tanh')
        self.blstm_layer = tf.keras.layers.Bidirectional(self.lstm_layer, backward_layer=self.lstm_layer_back)
        self.lstm_layer2 = tf.keras.layers.LSTM(n_lstm_units,
                                               return_sequences=True,
                                               activation='tanh')
        self.lstm_layer_back2 = tf.keras.layers.LSTM(n_lstm_units,
                                                    return_sequences=True,
                                                    go_backwards=True,
                                                    activation='tanh')
        self.blstm_layer2 = tf.keras.layers.Bidirectional(self.lstm_layer2, backward_layer=self.lstm_layer_back2)
        self.dense_layer = tf.keras.layers.Dense(n_dense_units)

    def call(self, x, training=None, mask=None):
        x = self.conv_layer(x)
        x = self.blstm_layer(x)
        x = self.blstm_layer2(x)
        x = self.dense_layer(x)
        return x
