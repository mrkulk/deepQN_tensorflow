import cv2
import numpy as np
import copy
import sys
from ale_python_interface import ALEInterface


class emulator:
	def __init__(self, rom_name):
		self.ale = ALEInterface()
		self.max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode");
		self.ale.setInt("random_seed",123)
		self.ale.loadROM('/home/tejas/Documents/MIT/alewrap/roms/' + rom_name )
		self.legal_actions = ale.getMinimalActionSet()
		self.screen_width,self.screen_height = ale.getScreenDims()
		print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))

	def next(self, action_indx):
		reward = ale.act(action_indx)
    # numpy_surface = np.frombuffer(screen.get_buffer(),dtype=np.int32)
    # ale.getScreenRGB(numpy_surface)

if __name__ == "__main__":
	engine = emulator('montezuma_revenge.bin')