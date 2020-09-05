import tensorflow as tf
from model import ASR
from generator import DataGenerator, ALPHABET
from train import train

"""
Trains a model with Albayzin data and saves it.
"""

DATA = 'C:/Users/Asier/PycharmProjects/ASR1/DATOS/audio/wav/albayzin_train'
LABELS = 'C:/Users/Asier/PycharmProjects/ASR1/DATOS/audio/text/text_albayzin_train'
MODEL = 'C:/Users/Asier/PycharmProjects/ASR1/Modelo3'

gen = DataGenerator()
gen.load_data(DATA, LABELS)

model = ASR(200, 11, 2, 'valid', 200, len(ALPHABET))
optimizer = tf.keras.optimizers.Adam()

train(model, optimizer, gen, 16, 10)
tf.keras.models.save_model(model, MODEL)
