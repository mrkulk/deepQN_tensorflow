import cv2
import numpy as np
import copy
import sys
from ale_python_interface import ALEInterface
import scipy.misc
import pygame


class emulator:
	def __init__(self, rom_name, vis):
		self.ale = ALEInterface()
		self.max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode");
		self.ale.setInt("random_seed",123)
		self.ale.loadROM('/home/tejas/Documents/MIT/alewrap/roms/' + rom_name )
		self.legal_actions = self.ale.getMinimalActionSet()
		self.screen_width,self.screen_height = self.ale.getScreenDims()
		print("width/height: " +str(self.screen_width) + "/" + str(self.screen_height))
		if vis:
			pygame.init()
			self.screen = pygame.display.set_mode((self.screen_width,self.screen_height))
			pygame.display.set_caption("Arcade Learning Environment Random Agent Display")
			pygame.display.flip()
		self.vis = vis

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
		if self.vis:
			numpy_surface = np.frombuffer(self.screen.get_buffer(),dtype=np.int8)
			self.ale.getScreenRGB(numpy_surface)
			pygame.display.flip()
		return nextstate, reward, self.ale.game_over()



if __name__ == "__main__":
	engine = emulator('montezuma_revenge.bin')
	engine.next(0)
