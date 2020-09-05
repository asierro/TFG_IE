import tensorflow as tf
import numpy as np
import librosa
from string import ascii_uppercase
import random
import os
from concurrent.futures import ThreadPoolExecutor, wait

"""
Defines a class that is used to featurize audio clips, and provide
them to the network in batches for training or testing.
"""

RNG_SEED = 123

SPACE_TOKEN = ' '
END_TOKEN = '>'
BLANK_TOKEN = '%'
ALPHABET = list(ascii_uppercase) + list('ÁÉÍÓÚÑ') + [SPACE_TOKEN, END_TOKEN, BLANK_TOKEN]


def generate_target_output_from_text(target_text):
    """
    Target output is an array of indices for each character in your string.
    The indices comes from a mapping that will be used while decoding the ctc output.
    :param target_text: (str) target string
    :return: list of indices for each character in the string
    """
    char_to_index = {}
    for idx, char in enumerate(ALPHABET):
        char_to_index[char] = idx

    y = []
    for char in target_text:
        y.append(char_to_index[char])
    return y


class DataGenerator(object):
    """
    Class that make generators to iterate through data batches.
    """
    def __init__(self, sampling=8000):
        """
        :param sampling: new sampling rate for the signals.
        :param rng:
        """
        self.sampling = sampling
        self.rng = random.Random(RNG_SEED)

    @staticmethod
    def read_data(data_path, labels_file):
        """
        Save audio file paths and their corresponding transcription in lists.
        :param data_path: path to folder with the .wav files.
        :param labels_file: file with the transcriptions of the .wav files in data_path, where each line has the format:
        wav_file transcprition
        :return: (keys, labels) tuple where keys is a list with the paths to the audio files and labels is a list with
        the transcriptions corresponding to the same index.
        """
        labels = []
        keys = []
        for line in open(labels_file, encoding='utf-8'):
            split = line.strip().split()
            file_id = split[0]
            label = ' '.join(split[1:]) + END_TOKEN  # end token necessary to apply lm to last word
            audio_file = os.path.join(data_path, file_id) + '.wav'
            keys.append(audio_file)
            labels.append(label)

        return keys, labels

    def load_data(self, data_path, labels_file):
        """
        Load audio file paths and label data.
        :param data_path: path to folder with the .wav files.
        :param labels_file: file with the transcriptions of the .wav files in data_path, where each line has the format:
        wav_file transcription.
        """
        audio_paths, texts = self.read_data(data_path, labels_file)
        self.audio_paths = audio_paths
        self.label_texts = texts

    @staticmethod
    def create_spectrogram(signals):
        """
        Function to create spectrogram from signals loaded from an audio file.
        :param signals:
        :return:
        """
        stfts = tf.signal.stft(signals, frame_length=200, frame_step=80, fft_length=256)
        spectrograms = tf.math.pow(tf.abs(stfts), 0.5)  # should be 2. instead of 0.5
        return spectrograms

    def generate_input_from_audio_file(self, path_to_audio_file, resample_to=None):
        """
        Function to create input for our neural network from an audio file.
        The function loads the audio file using librosa, resamples it, and creates spectrogram form it
        :param path_to_audio_file: path to the audio file
        :param resample_to: (int) new sampling rate for the signals.
        :return: spectrogram corresponding to the input file
        """
        # load the signals and resample them
        resample_to = self.sampling if resample_to is None else resample_to
        signal, sample_rate = librosa.core.load(path_to_audio_file)
        if signal.shape[0] == 2:
            signal = np.mean(signal, axis=0)
        signal_resampled = librosa.core.resample(signal, sample_rate, resample_to)

        # create spectrogram
        x = DataGenerator.create_spectrogram(signal_resampled)

        # normalisation
        means = tf.math.reduce_mean(x, 1, keepdims=True)
        stddevs = tf.math.reduce_std(x, 1, keepdims=True)
        x = tf.math.divide_no_nan(tf.subtract(x, means), stddevs)
        return x

    def prepare_batch(self, audio_paths, texts):
        """ Featurize a batch of audio, zero pad them and return a dictionary
        Params:
            audio_paths (list(str)): List of paths to audio files
            texts (list(str)): List of texts corresponding to the audio files
        Returns:
            dict: See below for contents
        """
        assert len(audio_paths) == len(texts), \
            "Inputs and outputs to the network must be of the same number"
        # Features is a list of (timesteps, feature_dim) arrays
        features = [self.generate_input_from_audio_file(a) for a in audio_paths]
        indices = [generate_target_output_from_text(t) for t in texts]
        input_lengths = [f.shape[0] for f in features]
        label_lengths = [len(indices[i]) for i in range(len(indices))]
        max_input = max(input_lengths)
        max_label = max(label_lengths)
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        # Pad inputs and labels so that they are all the same length
        x = np.zeros((mb_size, max_input, feature_dim))
        y = np.zeros((mb_size, max_label))
        for i in range(mb_size):
            feat = features[i]
            x[i, :feat.shape[0], :] = feat
            idx = indices[i]
            y[i, :len(idx)] = idx
        # Convert to tensor for tf.nn.ctc_loss
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y, dtype=tf.int32)
        return {
            'x': x,  # 0-padded features of shape(batch_size, timesteps, feature_dim)
            'y': y,  # 0-padded labels of shape (batch_size, (integer sequences)
            'texts': texts,  # list(str) Original texts
            'input_lengths': input_lengths,  # list(int) Length of each input
            'label_lengths': label_lengths  # list(int) Length of each label
        }

    def get_generator(self, batch_size, shuffle=True):
        audio_paths = self.audio_paths
        texts = self.label_texts
        num_samples = len(audio_paths)

        def generator():
            if shuffle:
                temp = list(zip(audio_paths, texts))
                self.rng.shuffle(temp)
                x, y = list(zip(*temp))
            else:
                x = audio_paths
                y = texts

            pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
            future = pool.submit(self.prepare_batch,
                                 x[:batch_size],
                                 y[:batch_size])
            for offset in range(batch_size, num_samples, batch_size):
                wait([future])
                batch = future.result()
                # While the current batch is being consumed, prepare the next
                future = pool.submit(self.prepare_batch,
                                     x[offset: offset + batch_size],
                                     y[offset: offset + batch_size])
                yield batch

        return generator()
