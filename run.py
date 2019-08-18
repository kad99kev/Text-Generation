import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tf.logging.set_verbosity(tf.logging.ERROR)

new_model = tf.keras.models.load_model('model_text_gen.h5')

import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def generate_text(seed_text, num_words, model, max_sequence_length):
    for _ in range(num_words):
        generate_random = False
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        if len(token_list) == 0:
            generate_random = True
        token_list = pad_sequences([token_list],
                                   maxlen=max_sequence_length - 1,
                                             padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        if generate_random:
            pred_list = np.argsort(-model.predict_proba(token_list))[0]
            temp = np.random.randint(low=0, high=10)
            predicted = pred_list[0]
            print(temp)
            print(predicted)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text.title()

generate_text("jesus", 10, new_model, new_model.layers[0].input_shape[1] + 1)
generate_text("lord", 10, new_model, new_model.layers[0].input_shape[1] + 1)

while True:
    choice = int(input('Enter\n1. To enter text\n2. To quit\n->'))
    if choice == 1:
        seed_text = input('Enter text: ')
        num_words = int(input("Enter number of words to be generated: "))
        print(generate_text(seed_text.lower(), num_words, new_model, new_model.layers[0].input_shape[1] + 1))
    elif choice == 2:
        break
    else:
        print("Invalid choice")