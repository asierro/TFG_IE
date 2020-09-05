import tensorflow as tf
from generator import DataGenerator

"""
Defines training function for the netwokr using CTC loss.
"""

def compute_ctc_loss(logits, labels, logit_length, label_length):
    """
    Function to compute CTC loss.
    Note: tf.nn.ctc_loss applies log softmax to its input automatically
    :param logits: Logits from the output dense layer
    :param labels: Labels converted to array of indices
    :param logit_length: Array containing length of each input in the batch
    :param label_length: Array containing length of each label in the batch
    :return: array of ctc loss for each element in batch
    """
    return tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=False,
        unique=None,
        blank_index=-1,
        name=None
    )


def train_sample(x, y, optimizer, model):
    """
    Function to perform forward and backpropagation on one batch.
    :param x: one batch of input
    :param y: one batch of target
    :param optimizer: optimizer
    :param model: object of the ASR class
    :return: loss from this step
    """
    with tf.GradientTape() as tape:
        logits = model(x)
        labels = y
        logits_length = [logits.shape[1]]*logits.shape[0]
        labels_length = [labels.shape[1]]*labels.shape[0]
        loss = compute_ctc_loss(logits, labels, logit_length=logits_length, label_length=labels_length)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train(model, optimizer, data_gen, batch_size, epochs):
    """
    Function to train the model for given number of epochs.
    :param model: object of class ASR
    :param optimizer: optimizer
    :param data_gen: loaded DataGenerator object
    :param batch_size:
    :param epochs:
    :return: None
    """
    for step in range(0, epochs):
        generator = data_gen.get_generator(batch_size)
        for i, batch in enumerate(generator):
            x = batch['x']
            y = batch['y']
            loss = train_sample(x, y, optimizer, model)
            if i % 10 == 0:
                print('Epoch {}, Loss: {}'.format(step + 1, loss))
