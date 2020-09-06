import tensorflow as tf
import numpy as np
from scipy.special import softmax
import itertools
from model import ASR
from generator import *
import arpa
import os
from prefix_beam_search import prefix_beam_search
from string import ascii_uppercase

"""
Makes a prediction using prefix beam search decoding.
"""

SPACE_TOKEN = ' '
END_TOKEN = '>'
BLANK_TOKEN = '%'
ALPHABET = list(ascii_uppercase) + list('ÁÉÍÓÚÑ') + [SPACE_TOKEN, END_TOKEN, BLANK_TOKEN]

print('Loading LM model...')
# MODELS = arpa.loadf('C:/Users/Asier/PycharmProjects/ASR1/DATOS/text/milm.arpa', encoding='utf-8')
MODELS = arpa.loadf('C:/Users/Asier/PycharmProjects/ASR1/DATOS/albayzin_train.lm', encoding='utf-8')
LM = MODELS[0]  # ARPA files may contain several models.
print('Loaded LM model.')


def lm_prob(string):
    """
    Same as LM.p but returns 0.0 instead of raising KeyError when words appear that are not in the corpus.
    """
    try:
        return LM.p(string)
    except KeyError:
        return 0.


#VARIABLES = 'C:/Users/Asier/PycharmProjects/ASR1/Modelo/variables/variables'
VARIABLES = 'C:/Users/Asier/PycharmProjects/ASR1/Modelo3/variables/variables'
gen = DataGenerator()
AUDIO = 'C:/Users/Asier/PycharmProjects/ASR1/DATOS/audio/wav/albayzin_test/xdfa0100.wav'

model = ASR(200, 11, 2, 'valid', 200, len(ALPHABET))
model.load_weights(VARIABLES)

'''
x = gen.generate_input_from_audio_file(AUDIO)
x = tf.expand_dims(x, axis=0)  # converting input into a batch of size 1

# getting the ctc output
ctc_output = model(x)
ctc_output = tf.nn.softmax(ctc_output)
ctc_output = tf.squeeze(ctc_output).numpy()

# max_pos = int(np.argmax(ctc_output[:, 33]))  # position of maximum end_token probability
# maxv = ctc_output[max_pos, 33]
# 
# for t in range(max_pos+1, ctc_output[:, 0].size):
#     currv = ctc_output[t, 33]
#     currva = ctc_output[t, 0]
#     ctc_output[t] *= (1-maxv)/(1-currv-currva)
#     ctc_output[t, 33] = maxv
#     ctc_output[t, 0] = 0.0

# prefix beam search decoding
#lm_prob = lambda s: 1
output = prefix_beam_search(ctc_output, ALPHABET, BLANK_TOKEN, END_TOKEN, SPACE_TOKEN, lm=lm_prob)
print(output)
'''

labels_file = 'C:/Users/Asier/PycharmProjects/ASR1/DATOS/audio/text/text_albayzin_test'
data_path = 'C:/Users/Asier/PycharmProjects/ASR1/DATOS/audio/wav/albayzin_test'

labels = []
predictions = []
j = 0
for line in open(labels_file, encoding='utf-8'):
    split = line.strip().split()
    file_id = split[0]
    label = ' '.join(split[1:])
    audio_file = os.path.join(data_path, file_id) + '.wav'

    x = gen.generate_input_from_audio_file(audio_file)
    x = tf.expand_dims(x, axis=0)  # converting input into a batch of size 1
    ctc_output = model(x)
    ctc_output = tf.nn.softmax(ctc_output)
    ctc_output = tf.squeeze(ctc_output).numpy()
    output = prefix_beam_search(ctc_output, ALPHABET, BLANK_TOKEN, END_TOKEN, SPACE_TOKEN, lm=lm_prob)
    j += 1
    print(j)

    labels.append(label)
    predictions.append(output)

for i in range(len(labels)):
    print(labels[i])
    print(predictions[i])
    print('\n')
