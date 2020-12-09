# keras und numpy
from keras.callbacks import Callback
import numpy as np
import pandas as pd

class PriceDifference(Callback):
    def __init__(self, bow_test_description, x_test_variety, x_test_embed, y_test_price):
        self.bow_test_description = bow_test_description
        self.x_test_variety = x_test_variety
        self.x_test_embed = x_test_embed
        self.y_test_price = pd.np.array(y_test_price)

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