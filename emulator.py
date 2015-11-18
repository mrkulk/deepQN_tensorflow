import cv2
import numpy as np
import copy
import sys
from ale_python_interface import ALEInterface
import scipy.misc

class emulator:
	def __init__(self, rom_name):
		self.ale = ALEInterface()
		self.max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode");
		self.ale.setInt("random_seed",123)
		self.ale.loadROM('/home/tejas/Documents/MIT/alewrap/roms/' + rom_name )
		self.legal_actions = self.ale.getMinimalActionSet()
		self.screen_width,self.screen_height = self.ale.getScreenDims()
		print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))

	def next(self, action_indx):
		reward = self.ale.act(action_indx)	
		numpy_surface = np.zeros(self.screen_height*self.screen_width*3, dtype=np.uint8)
		self.ale.getScreenRGB(numpy_surface)
		nextstate = np.reshape(numpy_surface, (self.screen_height, self.screen_width, 3))
		return nextstate, reward


if __name__ == "__main__":
	engine = emulator('montezuma_revenge.bin')
	engine.next(0)
