# keras und diverses importieren
import os

import keras
import numpy as np
# dataset
from keras.datasets import imdb
# layers
from keras.layers import Input, Activation, Conv1D, MaxPool1D
from keras.layers import Embedding, LSTM, CuDNNLSTM, Dense              
# optimizer + loss
from keras.losses import binary_crossentropy
from keras.optimizers import Adagrad, Adam, RMSprop
# keras functional api
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence

# statische variablen
epochs = 3
learning_rate = 1e-3
batch_size = 64 # besser für weight update
vocab_size = 100000
embedding_length = 32
max_review_length = 500

def prepare_dataset():
    # dataset vorbereiten
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    # pad reviews zur gleichen länge
    x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

    # testset teilen für validation set
    x_size_test = int(len(x_test) * 0.66)
    x_val = x_test[x_size_test:]
    y_val = y_test[x_size_test:]

    x_test = x_test[:x_size_test]
    y_test = y_test[:x_size_test]

    print("Dataset:")
    print("Training: ", x_train.shape, y_train.shape)
    print("Test: ", x_test.shape, y_test.shape)
    print("Validation: ", x_val.shape, y_val.shape)

    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


def train(x_train, y_train, x_test, y_test, x_val, y_val):
    # model definieren
    review_input = Input(shape=(max_review_length,), name="review_input")
    # layers
    
    x = Embedding(vocab_size, embedding_length, input_length=max_review_length)(review_input)
    x = Conv1D(filters=32, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=2)(x)
    x = LSTM(100)(x)
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)

    # kompilieren
    opt = RMSprop(lr=learning_rate)
    model = Model(inputs=[review_input], outputs=[output])
    model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.summary()

    # training
    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)

    # score
    score = model.evaluate(x_test, y_test)
    print("Accuracy: %.2f%%" % (score[1] * 100))

    # model speichern
    model.save("review-predictor.npz")    

def predict():
    # reviews zum vorhersagen
    reviews = np.array([
        "I was amazed to see so many negative reviews; so many people are impossible to please. This movie was 2 1/2 hours long, but I could have sat there another 2 1/2 hours and not noticed. Thoroughly entertaining, and I love how the directors weren't afraid to take chances. I've read a lot of other user reviews that claim that there's no plot. Unless you're mentally handicapped or not paying attention because you're on your phone the entire movie, the plot is pretty clear, and decent in my opinion.",
        "this movie is so bad which is really hard",
        "You don't necessarily have to understand it immediately : The film will raise questions in you, such as : what is it to be a human being, is there some physical limitations to our humanity, how far could we be willing to go to determine knowledge, is there other dimensions that we can not access to, and above all: what is the nature of this intact and immutable bond that unites us to others wherever we are in the universe ? Is this bond only intelligible, or is it also tangible ? All these questions resonate in harmony in Nolan's Interstellar. Interstellar is itself a crescendo, increasing sensitivity and creativity. I use the term deliberately because it goes crescendo with the soundtrack by Hans Zimmer, which is one of the most beautiful music ever scored for a sci-fi movie. We are witnessing a perfect musical arrangement, a total symbiosis, a bit like the music of Gravity which had understood very well how to match the image and the rhythm of a sequence to its own musicality. Zimmer's crescendos are giving a new powerful breath to every new scene, whether it is in visually powerful & intense moments or in more intimate moments; it intrudes into our momentary feelings and sensations, and manages to extend them, sometimes almost to choking, before resting on the balance of the film frame along with our mind spell-bounded.",
        "This film is an example of having to many irons in the fire and not knowing what to do with them, which results in an over sentimental mess that is called Interstellar. Nolan finally shows clearly his hubris in that he thinks of himself as a cinematic mastermind, in fact he's just a Hollywood pet with cheap tricks. There is nothing wrong in getting the science incorrect, a competent filmmaker manages to work around the incoherence and still create a unified whole, Nolan on the other hand wants us to think that he understands the science behind Interstellar, but he also shows clearly that he doesn't put much trust in the audiences' intellect. This results in an over abundance of exposition which is there just to inform us of how stupid we are. The sentimentality, which is most of the time laughable, doesn't go along with the overall structure of the film because the plot is disorganized. The problem comes from the inability to transform a concept into a film without allowing the audience to interpret what they are seeing - instead Nolan tries to overwhelm us with his elitist vision and it comes out as a mess because he doesn't trust his audience, he's lack of vision doesn't come from him not having a concept but from the inability to mediate that concept through a film, in other words: he is a bad filmmaker.",
        "Interstellar is a film that wins the hearts of the audience not only with its sci-fi splendor, but also an emotional story that lies at its very heart. This film is not only about the discoveries, space exploration and the final frontier of mankind, but also about the relationship of father and daughter, who were in a difficult situation in life when one has to leave the other in the name of a goal that can not be underestimated. So, with what Nolan's genius unfolds before us this action is beyond praise. Combining the story, filled with not only real science fiction, but the true human values ​​and emotions, outstanding and very emotional performances, breathtaking visuals, epic and dramatic soundtrack, Christopher Nolan breathed the life into this film by his directing to create something truly masterpiece again.",
        "Having just finish watching this, I am both stunned and heartbroken. This is a gritty tale of life and the cause and effect of other peoples actions which can force someone into thinking their only way out is suicide. Don't dismiss this as yet another teen angst series; its not. Its an important piece of cinematography which about the darker side of life. If you have older teens, watch it with them. Make them see what it can really be like for some people and that their actions have consequences. It gets pretty hardcore at the end, making you think. Making you stop. Making you realise what what you should do is sometimes very different from what you actually do. You don't have to have handed someone the gun, sometimes its because you don't take it away from them too; doing nothing can be just as pivotal as causing the heartache. The acting is marvellous from these young adults depicting what its like trying to survive modern life as young adults and what they will, or wont, do to fit in."
    ])
    # reviews filtern das nur noch wörter da sind
    reviews_words_list = np.array([text_to_word_sequence(review) for review in reviews])
    print(reviews_words_list)
    # mapping von wörtern zu index laden von imdb
    imdb_word_indices = imdb.get_word_index()
    # mapping von wörtern zu index für unsere reviews generieren
    reviews_indices = []
    for reviews_words in reviews_words_list:
        indices = np.array([imdb_word_indices[word] if word in imdb_word_indices else 0 for word in reviews_words])
        print(indices)
        reviews_indices.append(indices)
    reviews_indices = np.array(reviews_indices)
    print(reviews_indices.shape, reviews_indices)
    # länge für unser netzwerk anpassen
    x = sequence.pad_sequences(reviews_indices, maxlen=max_review_length)
    print(x.shape, x)

    model = load_model("review-predictor.npz")
    predictions = model.predict(x)
    print(predictions)

def main():
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = prepare_dataset()
    
    if not os.path.exists("review-predictor.npz"):
        train(x_train, y_train, x_test, y_test, x_val, y_val)
    
    predict()

if __name__ == '__main__':
    main()
