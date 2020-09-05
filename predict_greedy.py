import tensorflow as tf
import itertools
from model import ASR
from generator import ALPHABET, BLANK_TOKEN, DataGenerator

"""
Makes a prediction using greedy decoding.
"""

#VARIABLES = 'C:/Users/Asier/PycharmProjects/ASR1/Modelo/variables/variables'
VARIABLES = 'C:/Users/Asier/PycharmProjects/ASR1/Modelo2/variables/variables'
gen = DataGenerator()
AUDIO = 'C:/Users/Asier/PycharmProjects/ASR1/DATOS/audio/wav/albayzin_test/xnfa0034.wav'

model = ASR(200, 11, 2, 'valid', 200, len(ALPHABET))
model.load_weights(VARIABLES)

x = gen.generate_input_from_audio_file(AUDIO)
x = tf.expand_dims(x, axis=0)  # converting input into a batch of size 1

# getting the ctc output
ctc_output = model(x)
ctc_output = tf.nn.log_softmax(ctc_output)

# greedy decoding
output_text = ''
for timestep in ctc_output[0]:
    output_text += ALPHABET[tf.math.argmax(timestep)]
print(output_text)
output_text = ''.join(c[0] for c in itertools.groupby(output_text)).replace(BLANK_TOKEN, '')
print(output_text)
