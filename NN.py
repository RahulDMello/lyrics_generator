# RNN IMPLEMENTATION
import pickle
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Activation, Dense
import numpy as np
import time
from pickle_utils import *
import matplotlib.pyplot as plt
  
each_len = 32
n_input = 24
n_output = 1 * each_len
  
# config = {num_layers: , hidden_state: []}   
def nn_model(config):
	model = Sequential()
	model.add(LSTM(config['hidden_state'][0], return_sequences=True, input_shape=(n_input, each_len)))
	model.add(Dropout(0.1))
	for i in range(1,config['num_layers'] - 1):
		model.add(LSTM(config['hidden_state'][i], return_sequences=True))
		model.add(Dropout(0.1))
	if(config['num_layers'] > 2):
		model.add(LSTM(config['hidden_state'][config['num_layers']-1], return_sequences=False))
		model.add(Dropout(0.1))
	model.add(Dense(n_output))
	model.add(Activation('softmax'))
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
	return model
    
	
def train_nn(X, Y, hm_epochs, config):
	model = nn_model(config)
	history = model.fit(np.array(X), np.array(Y), batch_size=256, epochs=hm_epochs)
	model_json = model.to_json()
	with open("model/eminem.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("eminem.h5")
	print("saved model to model/eminem.json")
	return history
 

    
# BASIC VARIABLE AND META VARIABLE INITIALIZE
dict = load_obj("dataset/eminem")
input_train = dict["input"]
output_train = dict["output"]
print(np.shape(input_train))
print(np.shape(output_train))
n_epochs = 100

config = {'num_layers': 5, 'hidden_state': [256, 256, 512, 512, 1024], 'n_output': 89, 'n_examples': len(input_train), 'batch_size': 256, 'learning_rate': 0.001}

start_time = time.time()
history = train_nn(input_train, output_train, n_epochs, config)
print("--- %s seconds ---" % (time.time() - start_time))
plt.plot(history.history["loss"])
plt.savefig("loss.png")
plt.pause(5)





















