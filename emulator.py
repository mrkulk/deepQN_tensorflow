import numpy as np
import copy
import sys
from ale_python_interface import ALEInterface
import scipy.misc

class emulator:
	def __init__(self, rom_name, vis):
		if vis:
			import cv2
		self.ale = ALEInterface()
		self.max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode");
		self.ale.setInt("random_seed",123)
		self.ale.setInt("frame_skip",4)
		self.ale.loadROM('roms/' + rom_name )
		self.legal_actions = self.ale.getMinimalActionSet()
		self.action_map = dict()
		for i in range(len(self.legal_actions)):
			self.action_map[self.legal_actions[i]] = i

		# print(self.legal_actions)
		self.screen_width,self.screen_height = self.ale.getScreenDims()
		print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
		self.vis = vis
		if vis: 
			cv2.startWindowThread()
			cv2.namedWindow("preview")

	def get_image(self):
		numpy_surface = np.zeros(self.screen_height*self.screen_width*3, dtype=np.uint8)
		self.ale.getScreenRGB(numpy_surface)
		image = np.reshape(numpy_surface, (self.screen_height, self.screen_width, 3))
		return image

	def newGame(self):
		self.ale.reset_game()
		return self.get_image()

	def next(self, action_indx):
		reward = self.ale.act(action_indx)	
		nextstate = self.get_image()
		# scipy.misc.imsave('test.png',nextstate)
		if self.vis:
			cv2.imshow('preview',nextstate)
		return nextstate, reward, self.ale.game_over()



if __name__ == "__main__":
	engine = emulator('montezuma_revenge.bin')
	engine.next(0)
