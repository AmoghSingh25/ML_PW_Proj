import numpy as np
import random
import copy
import cv2 as cv
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
import time
from tqdm.notebook import tqdm

from config import *

from utils import get_coast_noise, periodic_kernel, white_kernel
from visualization import display_map

class simulation:
    def __init__(self, wave_freq, wave_speed, wave_decay, wave_cutoff, wave_retreat_coeff, wave_height, sand_pull, ground_pull, water_decay, wave_vol, wave_amplitude, wave_spread, obstacle_coords=[], dim_map=100, erosion_speedup=1):
        global WAVE_FREQ, WAVE_SPEED, WAVE_DECAY, WAVE_CUTOFF, WAVE_RETREAT_COEFF, WAVE_HEIGHT
        global SAND_PULL, GROUND_PULL, WATER_DECAY
        global GROUND_COLOR, SAND_COLOR, WATER_COLOR
        
        self.GROUND_COLOR = GROUND_COLOR
        self.SAND_COLOR = SAND_COLOR
        self.WATER_COLOR = WATER_COLOR

        self.FRAC_GROUND, self.FRAC_SAND = FRAC_GROUND, FRAC_SAND
        
        self.WAVE_FREQ = wave_freq
        self.WAVE_SPEED = wave_speed
        self.WAVE_DECAY = wave_decay
        self.WAVE_CUTOFF = wave_cutoff
        self.WAVE_RETREAT_COEFF = wave_retreat_coeff
        self.WAVE_HEIGHT = wave_height
        self.WAVE_VOL = wave_vol
        self.WAVE_AMPLITUDE = wave_amplitude
        self.WAVE_SPREAD = wave_spread
        self.EROSION_SPEEDUP = 1

        self.waves = []
        self.wave_speeds = []
        self.wave_dirs = []
        self.wave_vol = []

        self.WIND_DIR = 0 # Angle relative to shore, 0 - perpendicular to shore, 90 - parallel to shore, +ve Moving above, -ve Moving below

        self.obstacle_coords = obstacle_coords
        self.obstacle_map = np.zeros((dim_map,dim_map))
        self.place_obstacles()

        self.SAND_PULL = sand_pull
        self.GROUND_PULL = ground_pull
        self.WATER_DECAY = water_decay


        self.DIM_MAP = dim_map
    
    def get_coast_coords(self, coast_map, check_water = False, limit=0):
        coast_coords = []
        for i in range(self.DIM_MAP):
            if not check_water:
                if max(coast_map[i,:,1]) > limit:
                    edge_coord = max(np.where(coast_map[i,:,1] > limit)[0])
                else:
                    edge_coord = 0
            else:
                if min(coast_map[i,:,2]) > limit:
                    edge_coord = min(np.where(coast_map[i,:,2] > limit)[0])
                else:
                    edge_coord= 0 
                
            coast_coords.append((i,edge_coord))
        return np.array(coast_coords)


    def plot_coast(self, coords, label=''):
        plt.plot(coords[:,1],coords[:,0], label=label)
        plt.xlim(0,100)
        plt.ylim(0,100)
    
    def calculate_wave_pos(self, waves):
        return np.clip(waves,0,self.DIM_MAP-1).astype(np.int16)

    def get_wave(self, scaling_factor=3, period=1, noise_level=0.01, lengthScale=1, varSigma=1):
        wave = get_coast_noise(scaling_factor, period, noise_level)
        min_idx = min(wave)
        if min_idx < 0:
            wave = wave - min_idx
        wave = self.DIM_MAP - wave
        wave = self.calculate_wave_pos(wave)
        return wave
        
    def place_obstacles(self):
        for coord in self.obstacle_coords:
            y, x = int(coord[0]), int(coord[1])
            # Create a vertical line of obstacles extending 5 points up and down
            for offset in range(-5, 6):
                new_y = y + offset
                # Ensure new_y is within valid bounds of the obstacle_map
                if 0 <= new_y < self.obstacle_map.shape[0]:
                    self.obstacle_map[new_y, x] = 1


    def get_coast(self, coast_inp):
        temp_coast = copy.deepcopy(coast_inp)

        for i in range(len(self.rand_coast)):
            if self.rand_coast[i] < 0:
                temp_coast[i, int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND))+int(self.rand_coast[i]):,2] = 100
                temp_coast[i, int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND))+int(self.rand_coast[i]):,1] = 0
            else:
                temp_coast[i, int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)):int(self.rand_coast[i])+int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)), 1] = 100
                temp_coast[i, int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)):int(self.rand_coast[i])+int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)), 2] = 0
        
            if self.rand_terrain[i] < 0:
                temp_coast[i, int(self.DIM_MAP*(self.FRAC_GROUND))+int(self.rand_terrain[i]):int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)), 1] = 100
                temp_coast[i, int(self.DIM_MAP*(self.FRAC_GROUND))+int(self.rand_terrain[i]):int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)), 0] = 0
            else:
                temp_coast[i, :int(self.DIM_MAP*(self.FRAC_GROUND))+int(self.rand_terrain[i]), 0] = 100
                temp_coast[i, :int(self.DIM_MAP*(self.FRAC_GROUND))+int(self.rand_terrain[i]), 1] = 0
        return temp_coast

    def move_sand(self, coast_inp):     
        global WAVE_DECAY, SAND_PULL, GROUND_PULL, WAVE_CUTOFF, WAVE_RETREAT_COEFF, WAVE_SPREAD, WATER_DECAY, WAVE_SPEED, WAVE_VOL

        self.waves = self.waves.astype(int)
        for wave_idx in range(len(self.waves)):
            for i in range(len(self.waves[0])):
                vert_idx = i 
                hor_idx = self.waves[wave_idx][i]
                curr_pix = copy.deepcopy(coast_inp[vert_idx, hor_idx])
                
                if self.wave_speeds[wave_idx][i] < self.WAVE_CUTOFF and self.wave_dirs[wave_idx][i] == 1:
                    self.wave_dirs[wave_idx][i] = -1
                    self.wave_speeds[wave_idx][i] = self.WAVE_VOL * self.WAVE_RETREAT_COEFF * self.EROSION_SPEEDUP
                    continue

                if self.wave_speeds[wave_idx][i] > self.WAVE_CUTOFF and self.wave_dirs[wave_idx][i] == 1:
                    left_pix = copy.deepcopy(coast_inp[vert_idx,hor_idx-1])
                    new_curr_pix = copy.deepcopy(curr_pix)
                    
                    if left_pix[0] == 0 and left_pix[1] == 0:
                        continue
                    
                    if self.obstacle_map[vert_idx, hor_idx] == 1:
                        self.wave_dirs[wave_idx][i] = 0
                        self.wave_speeds[wave_idx][i] = 0
                        continue
                    
                    energy = self.wave_speeds[wave_idx][i] * self.wave_vol[wave_idx][i] * self.EROSION_SPEEDUP
                    
                    
                    new_curr_pix[0] = max(curr_pix[0] - self.GROUND_PULL * curr_pix[0] * energy, 0)
                    new_curr_pix[1] = max(curr_pix[1] - self.SAND_PULL * curr_pix[1] * energy, 0)
                    
                    left_pix[0] = left_pix[0] + self.GROUND_PULL * curr_pix[0] * energy
                    left_pix[1] = left_pix[1] + self.SAND_PULL * curr_pix[1] * energy
                    
                    coast_inp[vert_idx, hor_idx] = new_curr_pix
                    coast_inp[vert_idx, hor_idx-1] = left_pix

                    self.wave_speeds[wave_idx][i] *= self.WAVE_DECAY
                    self.wave_vol[wave_idx][i] *= self.WAVE_DECAY

                    if i-1 > 0 and self.wave_dirs[wave_idx][i-1] == 0 and self.wave_speeds[wave_idx][i-1] > 0 and self.obstacle_map[vert_idx-1, hor_idx]!=0:
                        # print("REVIVE DEAD WAVE - 1")
                        self.wave_vol[wave_idx][i-1] = self.WAVE_SPREAD * self.wave_vol[wave_idx][i]
                        self.wave_dirs[wave_idx][i-1] = 1
                        self.wave_speeds[wave_idx][i-1] = (1 - self.WAVE_DECAY) * self.wave_speeds[wave_idx][i]
                        
                    
                    if i+1 < self.DIM_MAP and self.wave_dirs[wave_idx][i+1] == 0 and self.wave_speeds[wave_idx][i+1] > 0 and self.obstacle_map[vert_idx+1, hor_idx]!=0:
                        # print("REVIVE DEAD WAVE - 2")
                        self.wave_vol[wave_idx][i+1] = self.WAVE_SPREAD * self.wave_vol[wave_idx][i]
                        self.wave_dirs[wave_idx][i+1] = 1
                        self.wave_speeds[wave_idx][i+1] = (1 - self.WAVE_DECAY) * self.wave_speeds[wave_idx][i]
                    
                elif self.wave_dirs[wave_idx][i] < 0:
                    
                    if curr_pix[0] == 0 and curr_pix[1] == 0 and curr_pix[2] > 1 or self.wave_speeds[wave_idx][i] < self.WAVE_CUTOFF:
                        self.wave_speeds[wave_idx][i] == 0
                        self.wave_dirs[wave_idx][i] = 0
                        continue
                        

                    right_pix = coast_inp[vert_idx, hor_idx+1]
                    energy = self.wave_speeds[wave_idx][i] * self.WAVE_RETREAT_COEFF * self.EROSION_SPEEDUP
                    
                    new_curr_pix = copy.deepcopy(curr_pix)
                    new_curr_pix[0] = max(curr_pix[0] - self.GROUND_PULL * curr_pix[0] * energy, 0)
                    new_curr_pix[1] = max(curr_pix[1] - self.GROUND_PULL * curr_pix[1] * energy, 0)

                    right_pix[0] = right_pix[0] + curr_pix[0]*self.GROUND_PULL*energy
                    right_pix[1] = right_pix[1] + curr_pix[1]*self.GROUND_PULL*energy
                    
                    coast_inp[vert_idx, hor_idx] = new_curr_pix
                    coast_inp[vert_idx, hor_idx+1] = right_pix

                    self.wave_speeds[wave_idx][i] *= self.WAVE_DECAY

        return coast_inp
    
    def run_sim(self, num_timesteps, plots=False):

        coast_map = np.zeros((self.DIM_MAP, self.DIM_MAP,3), dtype=np.int16)
        coast_map[:,:int(self.DIM_MAP*self.FRAC_GROUND),0] = 100  ## Assign sand portion of the map
        coast_map[:,int(self.DIM_MAP*self.FRAC_GROUND):int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)),1] = 100  ## Assign sand portion of the map
        coast_map[:,int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)):, 2] = 100  ## Assign water portion of the map
        self.rand_coast = get_coast_noise(scaling_factor=self.DIM_MAP / 20, period=0.5, noise_level=1 / self.DIM_MAP)
        self.rand_terrain = get_coast_noise(scaling_factor=self.DIM_MAP/20, period=0.5, noise_level=1 / self.DIM_MAP)
        temp_coast_map = self.get_coast(coast_map)
        self.waves = np.array([self.get_wave(scaling_factor=3, period=self.WAVE_AMPLITUDE, noise_level=1/self.DIM_MAP).astype(np.int16)])
        self.wave_speeds = np.array([np.ones(self.waves[0].shape) * self.WAVE_SPEED])
        self.wave_dirs = np.array([np.ones(self.waves[0].shape)])
        self.wave_vol = np.array([self.WAVE_VOL * np.ones(self.waves[0].shape)])

        
        for t in tqdm(range(1,num_timesteps)):
            if t%(round(self.WAVE_FREQ,1)*100)==0 and (num_timesteps - t) > 10 :
                new_wave = self.get_wave(scaling_factor=10, period=self.WAVE_AMPLITUDE, noise_level=1/self.DIM_MAP).astype(np.int16).reshape(1,-1)
                self.waves = np.append(self.waves, new_wave, axis=0)
                self.wave_speeds = np.append(self.wave_speeds, np.array([np.ones(self.waves[0].shape) * self.WAVE_SPEED]), axis=0)
                self.wave_vol = np.append(self.wave_vol, np.array([self.WAVE_VOL * np.ones(self.waves[0].shape)]), axis=0)
                self.wave_dirs = np.append(self.wave_dirs, np.array([np.ones(self.waves[0].shape)]), axis=0)

                if plots:
                    display_map(temp_coast_map, self.waves, self.obstacle_coords)
                    plt.title("Time step = "+str(t))
                    plt.show()
            temp_coast_map = self.move_sand(copy.deepcopy(temp_coast_map))
            self.waves = (self.waves - self.wave_dirs).astype(int)

            if np.min(self.waves)<0:
                self.waves = self.calculate_wave_pos(self.waves)
                
            if plots:
                display_map(temp_coast_map, self.waves,self.obstacle_coords, openCV=False)
                plt.title("Time step = "+str(t))
                plt.show()

        before_map = self.get_coast(coast_map)
        coords1 = self.get_coast_coords(before_map, limit=0)
        before_erosion, after_erosion = 0, 0
        for i in coords1:
            before_erosion += np.sum(before_map[i[0],:i[1],1])
            after_erosion += np.sum(temp_coast_map[i[0],:i[1],1])

        return self.get_coast(coast_map), temp_coast_map, before_erosion, after_erosion