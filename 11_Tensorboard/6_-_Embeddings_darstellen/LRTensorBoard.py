from keras import backend as K
from keras.callbacks import TensorBoard

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', 
                histogram_freq=0, 
                batch_size=32, 
                write_graph=True, 
                write_grads=False, 
                write_images=False, 
                embeddings_freq=0, 
                embeddings_layer_names=None, 
                embeddings_metadata=None, 
                embeddings_data=None, 
                update_freq='epoch'):

        super().__init__(log_dir=log_dir,
                 histogram_freq=histogram_freq,
                 batch_size=batch_size,
                 write_graph=write_graph,
                 write_grads=write_grads,
                 write_images=write_images,
                 embeddings_freq=embeddings_freq,
                 embeddings_layer_names=embeddings_layer_names,
                 embeddings_metadata=embeddings_metadata,
                 embeddings_data=embeddings_data,
                 update_freq=update_freq)
    
    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs=logs)