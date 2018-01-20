import sys
from pickle_utils import *
import numpy as np
import os

each_len = 32

def one_hot_of(chr):
	res = np.zeros(each_len)
	if(ord(chr) >= 65 and ord(chr) <= 90):
		res[ord(chr)-65] = 1
	elif(ord(chr) >= 97 and ord(chr) <= 122 ):
		res[ord(chr)-97] = 1
	elif(chr == ','):
		res[26] = 1
	elif(chr == "'"):
		res[27] = 1
	elif(chr == '.'):
		res[28] = 1
	elif(chr == '\n'):
		res[29] = 1
	elif(chr == '-'):
		res[30] = 1
	elif(chr == ' '):
		res[31] = 1
	return res
	
def get_one_hot_array_from(filename):
	one_hot = []
	with open(filename) as f:
		while True:
			c = f.read(1)
			if not c:
				break
			one_hot.append(one_hot_of(c))
			
	return one_hot

def get_one_hot_split_array_from(filename, n_input, n_output):
	one_hot = get_one_hot_array_from(filename)
	input_array = []
	output_array = []
	for i in range(len(one_hot) - n_input - 1):
		input_array.append([])
		for j in range(n_input):
			input_array[i].append(one_hot[i+j])
	
	for i in range(n_input, len(one_hot) - n_output):
		k = i - n_input
		for j in range(n_output):
			output_array.append(one_hot[i+j])
	
	dict = {"input": np.array(input_array), "output": np.array(output_array)}
	return dict

def main():
	if(len(sys.argv) >= 4):
		filename = sys.argv[1]
		save_file = sys.argv[2]
		n_input = int(sys.argv[3])
		n_output = int(sys.argv[4])
		data = get_one_hot_split_array_from(filename, n_input, n_output)
		save_obj(data, save_file)
		print('pickle saved as',save_file)
		
	elif(len(sys.argv) > 2):
		filename = sys.argv[1]
		save_file = sys.argv[2]
		arr = np.array(get_one_hot_array_from(filename))
		save_obj(arr, save_file)
		
	else:
		print("Give a text file name and save file name as command line arguments!")
	
if("__name__" == "__main__") :
	main();
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	