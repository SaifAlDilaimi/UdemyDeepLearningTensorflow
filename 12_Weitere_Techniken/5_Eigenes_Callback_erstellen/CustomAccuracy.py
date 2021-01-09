# keras und numpy
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd

class CatDogRatio(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        model = self.model

        predictions = self.model.predict({
            'input_description': self.bow_test_description,
            'input_variety': self.x_test_variety,
            'input_description_pad': self.x_test_embed
        })
        
        price_diff = np.abs(predictions.flatten() - self.y_test_price)
        price_diff = np.sum(price_diff) / len(price_diff)
        print("Die Durchschnittliche Preisdifferenz für Epoche {:02d} ist {:.4f}".format(epoch+1, price_diff))