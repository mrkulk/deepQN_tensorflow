import cv2
import numpy as np
import copy
import sys
from ale_python_interface import ALEInterface


class emulator:
	def __init__(self, rom_name):
		self.opt = {
			'env': rom_name, 
			'game_path': '/home/tejas/Documents/MIT/alewrap/roms/',
			'useRGB': True
		}
		self.ale = ALEInterface()
		max_frames_per_episode = ale.getInt("max_num_frames_per_episode");
		ale.setInt("random_seed",123)
		ale.loadROM()
