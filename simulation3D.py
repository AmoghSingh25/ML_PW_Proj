import numpy as np
import random
import copy
import cv2 as cv
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
import time
from tqdm.notebook import tqdm


from config import *
from utils import white_kernel, periodic_kernel


class Simulation3D:
    def __init__(self, wave_freq, wave_speed, wave_decay, wave_cutoff, wave_retreat_coeff, wave_height, sand_pull, ground_pull, water_decay, wave_vol, wave_amplitude, wave_spread, obstacle_coords=[], dim_map=100, erosion_speedup=1):
        global WAVE_FREQ, WAVE_SPEED, WAVE_DECAY, WAVE_CUTOFF, WAVE_RETREAT_COEFF, WAVE_HEIGHT
        global SAND_PULL, GROUND_PULL, WATER_DECAY
        global GROUND_COLOR, SAND_COLOR, WATER_COLOR
        
        self.GROUND_COLOR = GROUND_COLOR
        self.SAND_COLOR = SAND_COLOR
        self.WATER_COLOR = WATER_COLOR
        
        self.DIM_MAP = dim_map
        

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

        self.EROSION_SPEEDUP = erosion_speedup

        self.SAND_LIM = 10
        self.HORIZON_HEIGHT = self.DIM_MAP // 2

        self.ENERGY_SCALE = 0.001

        self.waves = []
        self.wave_speeds = []
        self.wave_dirs = []
        self.wave_vol = []


        self.obstacle_coords = obstacle_coords
        self.obstacle_map = np.zeros((dim_map,dim_map,dim_map, 3))
        self.obstacle_height = self.HORIZON_HEIGHT + self.DIM_MAP//10

        self.SAND_PULL = sand_pull
        self.GROUND_PULL = ground_pull
        self.WATER_DECAY = water_decay



        self.coast_map = np.zeros((self.DIM_MAP, self.DIM_MAP,3), dtype=np.int16)
        self.coast_map[:,:int(self.DIM_MAP*self.FRAC_GROUND),0] = 100  ## Assign sand portion of the map
        self.coast_map[:,int(self.DIM_MAP*self.FRAC_GROUND):int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)),1] = 100  ## Assign sand portion of the map
        self.coast_map[:,int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)):, 2] = 100  ## Assign water portion of the map

        self.place_obstacles()


    def place_obstacles(self):
        for coord in self.obstacle_coords:
                y, x = int(coord[0]), int(coord[1])
                for offset in range(-5, 6):
                    new_y = y + offset
                    if 0 <= new_y < self.obstacle_map.shape[0]:
                        self.obstacle_map[new_y, x, :self.obstacle_height] = [1,1,1]

    def get_coast_noise(self, scaling_factor=3, period=1, noise_level=0.01, lengthScale=1, varSigma=1):
        x = np.linspace(0,self.DIM_MAP, self.DIM_MAP).reshape(-1,1)
        K = periodic_kernel(x,None,varSigma,period,lengthScale) + white_kernel(x,None,noise_level)
        mu = np.zeros(x.shape)
        f = scaling_factor* np.random.multivariate_normal(mu.flatten(), K, 1)
        return f[0]
    

    def get_wave(self, scaling_factor=3, period=1, noise_level=0.01, lengthScale=1, varSigma=1):
        wave = self.get_coast_noise(scaling_factor, period, noise_level, lengthScale, varSigma)
        min_idx = min(wave)
        if min_idx < 0:
            wave = wave - min_idx
        wave = self.DIM_MAP - wave
        wave = self.calculate_wave_pos(wave)
        return wave

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

    
    def height_function(self, x, coast_lim=50, coast_height=50, custom_func=None):
        if not custom_func is None:
            return custom_func(x)
    
        if (x<coast_lim):
            y = coast_height
        else:
            y = coast_height * np.exp((coast_lim-x)/15)
        return y

        
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

    def plot_3d_downsamp(self, red_dim = 20):
        dim_jump = self.DIM_MAP
        coast_map_3d_downsamp = np.zeros((red_dim, red_dim, red_dim,3))
        X, Y, Z = coast_map_3d_downsamp.shape[:3]
        voxel_array = np.ones((X, Y, Z), dtype=bool)
        
        for i in range(0,self.DIM_MAP,self.DIM_MAP//red_dim):
            for j in range(0,self.DIM_MAP,self.DIM_MAP//red_dim):
                for k in range(0,self.DIM_MAP,self.DIM_MAP//red_dim):
                    if np.sum(self.coast_map_3D[i,j,k]) == 0:
                        coast_map_3d_downsamp[i//(self.DIM_MAP // red_dim),j//(self.DIM_MAP // red_dim),k//(self.DIM_MAP // red_dim)] = [1,1,1]
                        voxel_array[i//(self.DIM_MAP // red_dim),j//(self.DIM_MAP // red_dim),k//(self.DIM_MAP // red_dim)] = False
                    else:
                        coast_map_3d_downsamp[i//(self.DIM_MAP // red_dim),j//(self.DIM_MAP // red_dim),k//(self.DIM_MAP // red_dim)] = self.coast_map_3D[i,j,k] // 100
        
        for i in range(len(self.obstacle_coords)):
            coords = self.obstacle_coords[i]
            voxel_array[coords[0]//(self.DIM_MAP//red_dim), coords[1]//(self.DIM_MAP//red_dim), :self.obstacle_height//(self.DIM_MAP//red_dim)] = True
            coast_map_3d_downsamp[coords[0]//(self.DIM_MAP//red_dim), coords[1]//(self.DIM_MAP//red_dim), :self.obstacle_height//(self.DIM_MAP//red_dim)] = [1,1,0]
        
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(211, projection='3d')
        
        ax.voxels(voxel_array, facecolors=coast_map_3d_downsamp)

        ax2 = fig.add_subplot(212, projection='3d')
        ax2.voxels(voxel_array, facecolors=coast_map_3d_downsamp, alpha=0.5)
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_zlim(0,red_dim)
        
        ax.view_init(30, 30)
        plt.show()

    def plot_3d(self):
        coast_map_3d_downsamp = np.zeros((self.DIM_MAP, self.DIM_MAP, self.DIM_MAP, 3))
        X, Y, Z = coast_map_3d_downsamp.shape[:3]
        voxel_array = np.ones((X, Y, Z), dtype=bool)
        
        for i in range(0,self.DIM_MAP):
            for j in range(0,self.DIM_MAP):
                for k in range(0,self.DIM_MAP):
                    if np.sum(self.coast_map_3D[i,j,k]) == 0:
                        coast_map_3d_downsamp[i,j,k] = [1,1,1]
                        voxel_array[i,j,k] = False
                    else:
                        voxel_array[i,j,k] = True
                        coast_map_3d_downsamp[i,j,k] = self.coast_map_3D[i,j,k] // np.max(self.coast_map_3D[i,j,k])

        for i in range(len(self.obstacle_coords)):
            coords = self.obstacle_coords[i]
            voxel_array[coords[0], coords[1], :self.obstacle_height] = True
            coast_map_3d_downsamp[coords[0], coords[1], :self.obstacle_height] = [1,1,0]
        
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(voxel_array, facecolors=coast_map_3d_downsamp, alpha=0.6)
        
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_zlim(0,self.DIM_MAP)
        
        ax.view_init(30, 30)
        plt.show()
        

    def _pre_sim_run(self):
        self.rand_coast = self.get_coast_noise(scaling_factor=self.DIM_MAP / 20, period=0.5, noise_level=1 / self.DIM_MAP)
        self.rand_terrain = self.get_coast_noise(scaling_factor=self.DIM_MAP/20, period=0.5, noise_level=1 / self.DIM_MAP)
        self.coast_map = self.get_coast(self.coast_map)
        self.coast_map_3D = np.zeros((self.DIM_MAP, self.DIM_MAP, self.DIM_MAP,3))
        self.force_maps = np.zeros((self.DIM_MAP,self.DIM_MAP,self.DIM_MAP,4)) # Dim 0 - Force magnitude, 1- Force direction
        
        
        for i in range(0, self.HORIZON_HEIGHT):
            self.coast_map_3D[:,:,i,:] = self.coast_map
        for i in range(self.DIM_MAP):
            layer_i = self.coast_map_3D[i,:,:]
            end_coast = max(np.where(layer_i[:,:,1] > 0)[0])
            for j in range(end_coast, self.DIM_MAP):
                depth_j = int(self.height_function(j,end_coast, self.HORIZON_HEIGHT))
                self.coast_map_3D[i,j,0:depth_j] = [0,100,0]
                self.coast_map_3D[i,j,depth_j:self.HORIZON_HEIGHT] = [0,0,100]
        self.coast_map_init = copy.deepcopy(self.coast_map_3D)


    def propagate_forces(self):
        new_force_map = np.zeros(self.force_maps.shape)
        ct = 0

        flag = False
        
        for i in range(self.DIM_MAP):
            for j in range(self.DIM_MAP):
                for k in range(self.DIM_MAP):
                    neighbour_cells, force_dirs = self.neighbours_3D(i,j,k)
                    force_dirs = -1 * np.array(force_dirs)
                    
                    for cell in range(len(neighbour_cells)):
                        idx = neighbour_cells[cell]
                        if np.sum(force_dirs[cell] * self.force_maps[idx[0],idx[1],idx[2],1:]) > 0:
                            # new_force_map[idx[0],idx[1],idx[2]] = np.array([])
                            if self.coast_map_3D[idx[0],idx[1],idx[2],1] > 0 or self.coast_map_3D[idx[0],idx[1],idx[2],0] > 0:
                                if self.force_maps[idx[0], idx[1], idx[2], 2] > 0:
                                    energy = self.WAVE_RETREAT_COEFF * self.force_maps[idx[0],idx[1],idx[2],0] * self.EROSION_SPEEDUP
                                    
                                    self.coast_map_3D[i,j,k,0] += energy * (self.GROUND_PULL * self.coast_map_3D[idx[0], idx[1], idx[2], 0])
                                    self.coast_map_3D[i,j,k,1] += energy * (self.SAND_PULL * self.coast_map_3D[idx[0], idx[1], idx[2], 1])
                                    
                                    self.coast_map_3D[idx[0],idx[1],idx[2],0] -= energy * (self.GROUND_PULL * self.coast_map_3D[idx[0], idx[1], idx[2], 0])
                                    self.coast_map_3D[idx[0],idx[1],idx[2],1] -= energy * (self.SAND_PULL * self.coast_map_3D[idx[0], idx[1], idx[2], 1])
                                    
                                    new_force_map[i,j,k,:] += self.WAVE_DECAY * self.force_maps[idx[0], idx[1], idx[2]]
                                    
                                else:
                                    energy = self.force_maps[idx[0],idx[1],idx[2],0] * self.EROSION_SPEEDUP
                                    
                                    self.coast_map_3D[i,j,k,0] += energy * (self.GROUND_PULL * self.coast_map_3D[idx[0], idx[1], idx[2], 0])
                                    self.coast_map_3D[i,j,k,1] += energy * (self.SAND_PULL * self.coast_map_3D[idx[0], idx[1], idx[2], 1])

                                    self.coast_map_3D[idx[0],idx[1],idx[2],0] -= energy * (self.GROUND_PULL * self.coast_map_3D[idx[0], idx[1], idx[2], 0])
                                    self.coast_map_3D[idx[0],idx[1],idx[2],1] -= energy * (self.SAND_PULL * self.coast_map_3D[idx[0], idx[1], idx[2], 1])
                                    
                                    new_force_map[i,j,k,:] += self.WAVE_DECAY * self.force_maps[idx[0], idx[1], idx[2]]
                                    
                            else:
                                new_force_map[i,j,k,:] += self.force_maps[idx[0], idx[1], idx[2]]
                                
                    if j-1 >=0 and new_force_map[i,j,k,2] == -1 and np.sum(self.obstacle_map[i,j-1,k]) > 1:
                        comb_force = np.sum(new_force_map[i,j,:,0])

                        height_raise = comb_force / (np.sum(self.coast_map_3D[i,j,:,2]) * 9.8)
                        new_force_map[i,j,k] = np.array([0,0,0,0])
                        
                        
                    elif j+1 < self.DIM_MAP and new_force_map[i,j,k,2] == 1 and np.sum(self.obstacle_map[i,j+1,k]) > 1:
                        new_force_map[i,j,k] = np.array([0,0,0,0])
                        
                        
                    if new_force_map[i,j,k,0] < self.WAVE_CUTOFF and new_force_map[i,j,k,0] > 0:
                        new_force_map[i,j,k] = np.array([self.WAVE_RETREAT_COEFF * self.get_wave_energy(k), new_force_map[i,j,k,1], 1 ,new_force_map[i,j,k,3]])
                    elif new_force_map[i,j,k,0] < 0 and np.abs(new_force_map[i,j,k,0]) < self.WAVE_CUTOFF:
                        new_force_map[i,j,k] = np.array([0,0,0,0])
                        

                        
        self.force_maps = new_force_map


    def plot_force_map(self):
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X,Y,Z = self.DIM_MAP,self.DIM_MAP,self.DIM_MAP
        voxel_array = np.ones((X, Y, Z), dtype=bool)
        colors = np.random.randint(0, 256, (X, Y, Z, 3)).astype(float)
        colors /= 255.0
        
        force_temp = np.zeros((X,Y,Z,4), dtype=np.float32)
    
        max_force = np.max(self.force_maps)    
    
        for i in tqdm(range(self.DIM_MAP)):
            for j in range(self.DIM_MAP):
                for k in range(self.DIM_MAP):
                    if self.force_maps[i,j,k][0] > 0:
                        voxel_array[i,j,k]=True
                        colors[i,j,k] = np.array([1,0,0])
                    elif self.force_maps[i,j,k][0] < 0:
                        voxel_array[i,j,k]=True
                        colors[i,j,k] = np.array([0,1,0])
                    else:
                        voxel_array[i,j,k]=False
                        colors[i,j,k] = np.array([0,0,0])
    
    
        ax.voxels(voxel_array, facecolors=colors)
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        
        # ax.view_init(30, 30)
        ax.view_init(30, 30)
        plt.show()
        plt.show()
        
    
    def get_wave_energy(self, wave_height):
        return (1.225 * 9.8 * wave_height**2 * self.WAVE_AMPLITUDE) / 8

    def generate_wave(self, wave_pos):
        for i in range(len(wave_pos)):
            max_depth_i = np.where(self.coast_map_3D[i,wave_pos[i],:,1] > self.SAND_LIM-1)[0]
            if len(max_depth_i) > 0:
                max_depth_i = max(max_depth_i)
            else:
                max_depth_i = self.HORIZON_HEIGHT
            
            for k in range(max_depth_i+1,self.HORIZON_HEIGHT+1):
                wave_energy = self.get_wave_energy(k - max_depth_i)
                self.force_maps[i,wave_pos[i],k] = [wave_energy,0,-1,0]

    
    def calculate_wave_pos(self, waves):
        return np.clip(waves,0,self.DIM_MAP-1).astype(np.int16)


    def neighbours_3D(self, i,j,k):
        dirs = [
            [0,0,1],
            [0,0,-1],
            [0,-1,0],
            [0,1,0],
            [-1,0,0],
            [1,0,0],
        ]
        neighbouring_cells = []
        forces_dirs = []
        for dim in dirs:
            new_pt = (i+dim[0], j+dim[1], k+dim[2])
            if new_pt[0] >= 0 and new_pt[0] < self.DIM_MAP and new_pt[1]>=0 and new_pt[1] < self.DIM_MAP and new_pt[2]>=0 and new_pt[2] < self.DIM_MAP:
                neighbouring_cells.append(new_pt)
                forces_dirs.append(dim)
        return neighbouring_cells, forces_dirs
        
    
    def run_sim(self, num_timesteps, plots=False):
        self._pre_sim_run()
    
        temp_coast_map = self.get_coast(self.coast_map)
        waves = np.array([self.get_wave(scaling_factor=10, period=self.WAVE_AMPLITUDE, noise_level=1/self.DIM_MAP).astype(np.int16)])
        
        self.generate_wave(waves[0])
        self.propagate_forces()
        
        for t in tqdm(range(1,num_timesteps)):
            if t%self.WAVE_FREQ == 0 and (num_timesteps - t) > 10 :
                new_wave = self.get_wave(scaling_factor=10, period=0.5, noise_level=1/self.DIM_MAP).astype(np.int16).reshape(1,-1)
                self.generate_wave(new_wave)
                waves = np.append(waves, new_wave, axis=0)
            self.propagate_forces()
        

        initial_sand = 0
        final_sand = 0
        idxs_init = list(set(list(zip(np.where(self.coast_map_init[:,:,:,1] > self.SAND_LIM)[0],np.where(self.coast_map_init[:,:,:,1] > self.SAND_LIM)[1]))))
        idxs_final = list(set(list(zip(np.where(self.coast_map_3D[:,:,:,1] > self.SAND_LIM)[0],np.where(self.coast_map_3D[:,:,:,1] > self.SAND_LIM)[1]))))

        for i in range(len(idxs_init)):
            idx = idxs_init[i]
            initial_sand += np.sum(self.coast_map_init[idx[0],idx[1],:,1])

        for i in range(len(idxs_final)):
            idx = idxs_final[i]
            final_sand += np.sum(self.coast_map_3D[idx[0],idx[1],:,1])

        return self.coast_map_init, self.coast_map_3D, initial_sand, final_sand