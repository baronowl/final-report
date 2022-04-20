

import keras
import numpy as np
import time
import matplotlib.pyplot as plt
import json
import os
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils.vis_utils import plot_model
from keras.callbacks import *
from sklearn.model_selection import train_test_split


	

class LossHistory(keras.callbacks.Callback):
	def init(self):
		self.losses = []
	
	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


# These code are modified from the code in this webiste:  https://blog.csdn.net/OliverkingLi/article/details/81214947

class TrainingMonitor(BaseLogger):

	def __init__(self, fig_path, model,
				 train_loss_path, test_loss_path, train_acc_path, test_acc_path, json_path=None, start_At=0):
		
		super(TrainingMonitor, self).__init__()
		self.fig_path = fig_path + "/xxx.png"
		self.json_path = json_path
		self.start_At = start_At
		self.model = model
		self.epochs = 0
		
		self.train_loss_path = train_loss_path
		self.test_loss_path = test_loss_path
		self.train_acc_path = train_acc_path
		self.test_acc_path = test_acc_path
	
	def on_train_begin(self, logs={}):
		print(self.json_path)
		self.H = {}
		if self.json_path is not None:
			if os.path.exists(self.json_path):
				self.H = json.loads(open(self.json_path).read())
				if self.start_At > 0:
					for k in self.H.keys():
						self.H[k] = self.H[k][:self.start_At]
	
	def on_epoch_end(self, epoch, logs=None):
		print(self.json_path)
		for (k, v) in logs.items():
			l = self.H.get(k, [])
			l.append(v)
			self.H[k] = l
		if self.json_path is not None:
			f = open(self.json_path, 'w')
			f.write(json.dumps(self.H))
			f.close()
		if len(self.H["loss"]) > 1:
			N = np.arange(0, len(self.H["loss"]))
			plt.style.use("ggplot")
			plt.figure()
			plt.plot(N, self.H["loss"], label="train_loss")
			write_txt_file(self.H["loss"], self.train_loss_path)
			plt.plot(N, self.H["val_loss"], label="val_loss")
			write_txt_file(self.H["val_loss"], self.test_loss_path)
			plt.plot(N, self.H["accuracy"], label="train_acc")
			write_txt_file(self.H["accuracy"], self.train_acc_path)
			plt.plot(N, self.H["val_accuracy"], label="val_accuracy")
			write_txt_file(self.H["val_accuracy"], self.test_acc_path)
			plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
			plt.xlabel("Epoch #")
			plt.ylabel("Loss/Accuracy")
			plt.legend()
			plt.savefig(self.fig_path)
			plt.close()

def write_txt_file(list_info, write_file_path):
	with open(write_file_path, "w") as f:
		for info in list_info:
			f.write(str(info) + "\n")