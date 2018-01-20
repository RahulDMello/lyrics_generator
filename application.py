import numpy as np
from keras.models import model_from_json
from text_to_one_hot_pickle import one_hot_of


each_len = 32
n_input = 24

def get_char_from_one_hot(one_hot):
	max = np.argmax(one_hot)
	if(max < 26):
		return chr(65+max)
	elif max == 26:
		return ','
	elif max == 27:
		return "'"
	elif max == 28:
		return '.'
	elif max == 29:
		return '\n'
	elif max == 30:
		return '-'
	
		
	
json_file = open("model/eminem.json")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("eminem.h5")



X = []
X.append(one_hot_of("y"))
X.append(one_hot_of("o"))
X.append(one_hot_of("u"))
X.append(one_hot_of(" "))
X.append(one_hot_of("b"))
X.append(one_hot_of("e"))
X.append(one_hot_of("t"))
X.append(one_hot_of("t"))
X.append(one_hot_of("e"))
X.append(one_hot_of("r"))
X.append(one_hot_of(" "))
X.append(one_hot_of("l"))
X.append(one_hot_of("o"))
X.append(one_hot_of("s"))
X.append(one_hot_of("e"))
X.append(one_hot_of(" "))
X.append(one_hot_of("y"))
X.append(one_hot_of("o"))
X.append(one_hot_of("u"))
X.append(one_hot_of("r"))
X.append(one_hot_of("s"))
X.append(one_hot_of("e"))
X.append(one_hot_of("l"))
X.append(one_hot_of("f"))

X = np.reshape(X,[1,n_input,each_len])

file = open("output.txt", "w")
file.write("you better lose yourself")
    
for _ in range(150):
    y = model.predict(X)
    ch = get_char_from_one_hot(y)    # try sample from a probability distribution
    file.write(ch)
    for i in range(n_input - 1):
        X[0][i] = X[0][i+1]
    X[0][n_input - 1] = one_hot_of(ch)
    print(ch)

file.close()


























