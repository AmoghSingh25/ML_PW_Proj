import numpy as np
import copy
from config import *
from utils import get_coast_noise
from visualization import display_map

class CoastalSimulation:
    def __init__(self):
        self.coast_map = np.zeros((DIM_MAP, DIM_MAP, 3), dtype=np.int16)
        self.initialize_coast_map()
        self.rand_coast = get_coast_noise(DIM_MAP, scaling_factor=DIM_MAP/20, period=0.5, noise_level=1/DIM_MAP)
        self.rand_terrain = get_coast_noise(DIM_MAP, scaling_factor=DIM_MAP/20, period=0.5, noise_level=1/DIM_MAP)

    def initialize_coast_map(self):
        self.coast_map[:,:int(DIM_MAP*FRAC_GROUND),0] = 100  # Assign ground portion
        self.coast_map[:,int(DIM_MAP*FRAC_GROUND):int(DIM_MAP*(FRAC_GROUND+FRAC_SAND)),1] = 100  # Assign sand portion
        self.coast_map[:,int(DIM_MAP*(FRAC_GROUND+FRAC_SAND)):,2] = 100  # Assign water portion

    def get_coast(self, coast_inp):
        temp_coast = copy.deepcopy(coast_inp)

        for i in range(len(self.rand_coast)):
            if self.rand_coast[i] < 0:
                temp_coast[i, int(DIM_MAP*(FRAC_GROUND+FRAC_SAND))+int(self.rand_coast[i]):,2] = 100
                temp_coast[i, int(DIM_MAP*(FRAC_GROUND+FRAC_SAND))+int(self.rand_coast[i]):,1] = 0
            else:
                temp_coast[i, int(DIM_MAP*(FRAC_GROUND+FRAC_SAND)):int(self.rand_coast[i])+int(DIM_MAP*(FRAC_GROUND+FRAC_SAND)), 1] = 100
                temp_coast[i, int(DIM_MAP*(FRAC_GROUND+FRAC_SAND)):int(self.rand_coast[i])+int(DIM_MAP*(FRAC_GROUND+FRAC_SAND)), 2] = 0
        
            if self.rand_terrain[i] < 0:
                temp_coast[i, int(DIM_MAP*(FRAC_GROUND))+int(self.rand_terrain[i]):int(DIM_MAP*(FRAC_GROUND+FRAC_SAND)), 1] = 100
                temp_coast[i, int(DIM_MAP*(FRAC_GROUND))+int(self.rand_terrain[i]):int(DIM_MAP*(FRAC_GROUND+FRAC_SAND)), 0] = 0
            else:
                temp_coast[i, :int(DIM_MAP*(FRAC_GROUND))+int(self.rand_terrain[i]), 0] = 100
                temp_coast[i, :int(DIM_MAP*(FRAC_GROUND))+int(self.rand_terrain[i]), 1] = 0
        return temp_coast

    @staticmethod
    def calculate_wave_pos(waves, pos):
        return np.clip(np.add(waves,pos), 0, DIM_MAP-1).astype(np.int16)

    def move_sand(self, coast_inp, waves, pos, wave_speeds):
        waves = self.calculate_wave_pos(waves, pos)
        for wave_idx in range(len(waves)):
            for i in range(len(waves[0])):
                vert_idx = i 
                hor_idx = waves[0][i]
                curr_pix = coast_inp[vert_idx, hor_idx]
                if wave_speeds[wave_idx][i] > WAVE_CUTOFF:
                    left_pix = coast_inp[vert_idx,hor_idx-1]
                    if left_pix[0] == 0 and left_pix[1] == 0:
                        continue
                    
                    wave_speeds[wave_idx][i] *= WAVE_DECAY
                    if wave_speeds[wave_idx][i] < 0.1:
                        wave_speeds[wave_idx][i] = -1
                        continue
                    new_curr_pix = copy.deepcopy(curr_pix)
                    new_curr_pix[0] = curr_pix[0] - curr_pix[0]* GROUND_PULL*wave_speeds[wave_idx][i]
                    new_curr_pix[1] = curr_pix[1] - curr_pix[1]* SAND_PULL*wave_speeds[wave_idx][i]
                    new_curr_pix[2] = curr_pix[2] - curr_pix[2] * WATER_DECAY
                    coast_inp[vert_idx, hor_idx] = new_curr_pix
                    
                    left_pix[0] = left_pix[0] + curr_pix[0]* GROUND_PULL*wave_speeds[wave_idx][i]
                    left_pix[1] = left_pix[1] + curr_pix[1]* SAND_PULL*wave_speeds[wave_idx][i]
                    left_pix[2] = left_pix[2] + curr_pix[2] * WATER_DECAY
                    coast_inp[vert_idx, hor_idx-1] = left_pix

                elif wave_speeds[wave_idx][i] < 0:
                    if curr_pix[0] == 0 and curr_pix[1] == 0 and curr_pix[2] > 1:
                        wave_speeds[wave_idx][i] == 0
                        continue
                    right_pix = coast_inp[vert_idx, hor_idx+1]
                    
                    new_curr_pix = copy.deepcopy(curr_pix)
                    new_curr_pix[0] = curr_pix[0] - WAVE_RETREAT_COEFF * curr_pix[0]*GROUND_PULL*wave_speeds[wave_idx][i]
                    new_curr_pix[1] = curr_pix[1] - WAVE_RETREAT_COEFF * curr_pix[1]*SAND_PULL*wave_speeds[wave_idx][i]
                    new_curr_pix[2] = curr_pix[2] - WAVE_RETREAT_COEFF * curr_pix[2] * WATER_DECAY
                    
                    right_pix[0] = right_pix[0] + WAVE_RETREAT_COEFF * curr_pix[0]*GROUND_PULL*wave_speeds[wave_idx][i]
                    right_pix[1] = right_pix[1] + WAVE_RETREAT_COEFF * curr_pix[1]*SAND_PULL*wave_speeds[wave_idx][i]
                    right_pix[2] = right_pix[2] + WAVE_RETREAT_COEFF * curr_pix[2] * WATER_DECAY

                    coast_inp[vert_idx, hor_idx] = new_curr_pix
                    coast_inp[vert_idx, hor_idx+1] = right_pix
                    
                    wave_speeds[wave_idx][i] *= WAVE_DECAY
        return coast_inp

    def run_sim(self, num_timesteps, wave_freq, wave_speed, wave_decay, wave_cutoff, 
                wave_retreat_coeff, wave_height, sand_pull, ground_pull, water_decay, plots=False):
        global WAVE_FREQ, WAVE_SPEED, WAVE_DECAY, WAVE_CUTOFF, WAVE_RETREAT_COEFF, WAVE_HEIGHT
        global SAND_PULL, GROUND_PULL, WATER_DECAY

        # Update global parameters
        WAVE_FREQ = wave_freq
        WAVE_SPEED = wave_speed
        WAVE_DECAY = wave_decay
        WAVE_CUTOFF = wave_cutoff
        WAVE_RETREAT_COEFF = wave_retreat_coeff
        WAVE_HEIGHT = wave_height
        SAND_PULL = sand_pull
        GROUND_PULL = ground_pull
        WATER_DECAY = water_decay

        temp_coast_map = self.get_coast(self.coast_map)
        waves = np.array([get_coast_noise(DIM_MAP, scaling_factor=10, period=0.5, noise_level=1/DIM_MAP).astype(np.int16)])
        pos = np.array([np.ones(waves[0].shape) * DIM_MAP])
        wave_speeds = np.array([np.ones(waves[0].shape) * WAVE_SPEED])

        for t in range(1, num_timesteps):
            if t % WAVE_FREQ == 0:
                new_wave = get_coast_noise(DIM_MAP, scaling_factor=10, period=0.5, noise_level=1/DIM_MAP).astype(np.int16).reshape(1,-1)
                waves = np.append(waves, new_wave, axis=0)
                pos = np.append(pos, np.ones(new_wave.shape) * DIM_MAP, axis=0)
                wave_speeds = np.append(wave_speeds, np.array([np.ones(waves[0].shape) * WAVE_SPEED]), axis=0)

                if plots:
                    display_map(temp_coast_map, self.calculate_wave_pos(waves, pos))

            temp_coast_map = self.move_sand(temp_coast_map, waves, pos, wave_speeds)
            pos = pos - wave_speeds
            if plots:
                display_map(temp_coast_map, self.calculate_wave_pos(waves, pos), openCV=False)

        sand_layer = temp_coast_map[:, :, 1]  # Extract sand layer
        total_sand = np.sum(sand_layer)  # Sum of all sand cells
        print(f"Total sand: {total_sand}")
        return temp_coast_map, total_sand