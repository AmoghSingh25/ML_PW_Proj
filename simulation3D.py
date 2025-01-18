import numpy as np
import random
import copy
import cv2 as cv
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
import time
from tqdm.notebook import tqdm

class Simulation3D:
    def __init__(self, wave_freq, wave_speed, wave_decay, wave_cutoff, wave_retreat_coeff, wave_height, sand_pull, ground_pull, water_decay, wave_vol, wave_amplitude, wave_spread, obstacle_coords=[], dim_map=100):
        self.GROUND_COLOR = [56,118,29]
        self.SAND_COLOR = [255,229, 153]
        self.WATER_COLOR = [41,134,204]

        self.FRAC_GROUND, self.FRAC_SAND = 0.3, 0.3
        
        self.WAVE_FREQ = wave_freq
        self.WAVE_SPEED = wave_speed
        self.WAVE_DECAY = wave_decay
        self.WAVE_CUTOFF = wave_cutoff
        self.WAVE_RETREAT_COEFF = wave_retreat_coeff
        self.WAVE_HEIGHT = wave_height
        self.WAVE_VOL = wave_vol
        self.WAVE_AMPLITUDE = wave_amplitude
        self.WAVE_SPREAD = wave_spread

        self.SAND_LIM = 60
        self.CENTRAL_PROP = 0.9
        self.HORIZON_HEIGHT = 50

        self.waves = []
        self.wave_speeds = []
        self.wave_dirs = []
        self.wave_vol = []

        self.WIND_DIR = 0 # Angle relative to shore, 0 - perpendicular to shore, 90 - parallel to shore, +ve Moving above, -ve Moving below

        self.obstacle_coords = obstacle_coords
        # self.obstacle_map = np.zeros((dim_map,dim_map))
        # self.place_obstacles()

        self.SAND_PULL = sand_pull
        self.GROUND_PULL = ground_pull
        self.WATER_DECAY = water_decay


        self.DIM_MAP = dim_map

        self.coast_map = np.zeros((self.DIM_MAP, self.DIM_MAP,3), dtype=np.int16)
        self.coast_map[:,:int(self.DIM_MAP*self.FRAC_GROUND),0] = 100  ## Assign sand portion of the map
        self.coast_map[:,int(self.DIM_MAP*self.FRAC_GROUND):int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)),1] = 100  ## Assign sand portion of the map
        self.coast_map[:,int(self.DIM_MAP*(self.FRAC_GROUND+self.FRAC_SAND)):, 2] = 100  ## Assign water portion of the map


    def white_kernel(self, x1, x2, varSigma):
        return varSigma*np.eye(x1.shape[0])
    
    def periodic_kernel(self, x1, x2, varSigma, period, lengthScale):
        if x2 is None:
            d = cdist(x1, x1)
        else:
            d = cdist(x1, x2)
        return varSigma*np.exp(-(2*np.sin((np.pi/period)*d)**2)/lengthScale**2)
    
    
    def get_coast_noise(self, scaling_factor=3, period=1, noise_level=0.01, lengthScale=1, varSigma=1):
        x = np.linspace(0,self.DIM_MAP, self.DIM_MAP).reshape(-1,1)
        K = self.periodic_kernel(x,None,varSigma,period,lengthScale) + self.white_kernel(x,None,noise_level)
        mu = np.zeros(x.shape)
        f = scaling_factor* np.random.multivariate_normal(mu.flatten(), K, 1)
        return f[0]

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

    def display_map(self, coast, waves_inp=[], waves_pos=[], stream=True, openCV=False):
        
        cv.namedWindow("Coast")
        
        img = np.zeros((coast.shape[0], coast.shape[1],3), dtype=int)
        coast_temp = copy.deepcopy(coast)
        coast_temp[:,:,0] = (coast_temp[:,:,0] / np.max(coast_temp[:,:,0])) * 255
        coast_temp[:,:,1] = (coast_temp[:,:,1] / np.max(coast_temp[:,:,1])) * 255
        coast_temp[:,:,2] = (coast_temp[:,:,2] / np.max(coast_temp[:,:,2])) * 255
        coast_temp = coast_temp[...,np.newaxis]
        img = np.array(self.GROUND_COLOR).reshape(1,1,-1)*coast_temp[:,:,0,:]   ## Ground Color
        img+= np.array(self.SAND_COLOR).reshape(1,1,-1)*coast_temp[:,:,1,:]   ## Sand Color
        img+= np.array(self.WATER_COLOR).reshape(1,1,-1)*coast_temp[:,:,2,:]   ## Water Color
        
    
        if len(waves_inp) > 0:
            for i in range(self.DIM_MAP):
                img[i,waves_inp[:,i]] = [0,0,255]
    
        if not openCV:
            plt.imshow(img)
    
    
    
        if openCV:
            img = img.astype(np.uint8)
            img_cv = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            img_cv = cv.resize(img_cv, (1000,1000))
            
            cv.imshow("Coast", img_cv)
            if stream:
                cv.waitKey(2)
            else:
                cv.waitKey(0)
                cv.destroyWindow("Coast")
                cv.waitKey(1)

    def height_function(self, x, coast_lim=50, coast_height=50, custom_func=None):
        if not custom_func is None:
            return custom_func(x)
    
        if (x<coast_lim):
            y = coast_height
        else:
            y = coast_height * np.exp((coast_lim-x)/30)
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

    def plot_3d_downsamp(self):
        red_dim = 20
        dim_jump = self.DIM_MAP // red_dim
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
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.voxels(voxel_array, facecolors=coast_map_3d_downsamp, alpha=0.5)
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_zlim(0,red_dim)
        
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

    def layer_movement(self, layer_idx):
        layer = self.coast_map_3D[:,layer_idx]
        force_map_layer_i = self.force_maps[:, layer_idx]
        force_map_layer_i_2 = self.force_maps[:, layer_idx-1]
        for i in range(self.DIM_MAP):
            for k in range(self.DIM_MAP):
                next_cells = self.get_connected_cells(i,layer_idx,k)
    
                for cell in next_cells:
                    if cell[0] == i and cell[2] == k:
                        # Central cell
                        force_map_layer_i_2[cell[0],cell[2],1:] = np.add(force_map_layer_i_2[cell[0],cell[2],1:] * force_map_layer_i_2[cell[0],cell[2],0], self.CENTRAL_PROP * force_map_layer_i[i,k,0] * force_map_layer_i[i,k,1:])
                        force_map_layer_i_2[cell[0],cell[2],0] = self.CENTRAL_PROP * force_map_layer_i[i,k,0]
                    else:
                        force_map_layer_i_2[cell[0],cell[2],1:] = np.add(force_map_layer_i_2[cell[0],cell[2],1:] * force_map_layer_i_2[cell[0],cell[2],0], (1-self.CENTRAL_PROP) * force_map_layer_i[i,k,0] * force_map_layer_i[i,k,1:])
                        force_map_layer_i_2[cell[0],cell[2],0] = ((1-self.CENTRAL_PROP) /len(next_cells)) * force_map_layer_i[i,k,0]
    
        self.force_maps[:,layer_idx] = force_map_layer_i
        self.force_maps[:,layer_idx-1] = force_map_layer_i_2

    
    def plot_force_map(self):
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X,Y,Z = 100,100,100
        voxel_array = np.ones((X, Y, Z), dtype=bool)
        colors = np.random.randint(0, 256, (X, Y, Z, 3)).astype(float)
        colors /= 255.0
        
        force_temp = np.zeros((X,Y,Z,4), dtype=np.float32)
    
        max_force = np.max(self.force_maps)
    
    
        for i in tqdm(range(100)):
            for j in range(100):
                for k in range(100):
                    if self.force_maps[i,j,k,0] > 0:
                        voxel_array[i,j,k]=True
                        colors[i,j,k] = np.array([self.force_maps[i,j,k,0] ,0,0])
                    else:
                        voxel_array[i,j,k]=False
                        colors[i,j,k] = np.array([0,0,0])
    
    
        ax.voxels(voxel_array, facecolors=colors)
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        
        # ax.view_init(30, 30)
        plt.show()

    def backward_movement(self):
        for i in range(self.DIM_MAP):
            self.layer_movement(i)
            
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
                self.force_maps[i,wave_pos[i],k] = [wave_energy,0,1,0]

    
    def calculate_wave_pos(self, waves, pos):
        return np.clip(np.add(waves,pos),0,self.DIM_MAP-1).astype(np.int16)
    
    def run_sim(self, num_timesteps, plots=False):
        self._pre_sim_run()
    
        temp_coast_map = self.get_coast(self.coast_map)
        waves = np.array([self.get_coast_noise(scaling_factor=10, period=self.WAVE_AMPLITUDE, noise_level=1/self.DIM_MAP).astype(np.int16)])
        self.generate_wave(waves[0])
        
        for t in tqdm(range(1,num_timesteps)):
            if t%self.WAVE_FREQ == 0 and (num_timesteps - t) > 10 :
                new_wave = self.get_coast_noise(scaling_factor=10, period=0.5, noise_level=1/self.DIM_MAP).astype(np.int16).reshape(1,-1)
                self.generate_wave(new_wave)
                waves = np.append(waves, new_wave, axis=0)
            self.backward_movement()
            # time.sleep(0.5)
        # return temp_coast_map

    def get_connected_cells(self, i,j,k):
        next_cells = []
        if j == 0:
            return []
        if i>0 and i<self.DIM_MAP-1:
            if k>0 and k < self.DIM_MAP-1:
                next_cells= [
                    [i,j-1,k+1],
                    [i,j-1,k],
                    [i,j-1,k-1],
                    [i-1,j-1,k+1],
                    [i-1,j-1,k],
                    [i-1,j-1,k-1],
                    [i+1,j-1,k+1],
                    [i+1,j-1,k],
                    [i+1,j-1,k-1]
                ]
            elif k==0:
                next_cells = [
                    [i,j-1,k+1],
                    [i,j-1,k],
                    [i-1,j-1,k+1],
                    [i-1,j-1,k],
                    [i+1,j-1,k+1],
                    [i+1,j-1,k]
                ]
            else:
                next_cells = [
                    [i,j-1,k],
                    [i,j-1,k-1],
                    [i-1,j-1,k],
                    [i-1,j-1,k-1],
                    [i+1,j-1,k],
                    [i+1,j-1,k-1]
                ]
        elif i==0:
            if k>0 and k < self.DIM_MAP-1:
                next_cells= [
                    [i,j-1,k+1],
                    [i,j-1,k],
                    [i,j-1,k-1],
                    [i+1,j-1,k+1],
                    [i+1,j-1,k],
                    [i+1,j-1,k-1]
                ]
            elif k==0:
                next_cells = [
                    [i,j-1,k+1],
                    [i,j-1,k],
                    [i+1,j-1,k+1],
                    [i+1,j-1,k]
                ]
            else:
                next_cells = [
                    [i,j-1,k],
                    [i,j-1,k-1],
                    [i+1,j-1,k],
                    [i+1,j-1,k-1]
                ]
        else:
            if k>0 and k < self.DIM_MAP-1:
                next_cells= [
                    [i,j-1,k+1],
                    [i,j-1,k],
                    [i,j-1,k-1],
                    [i-1,j-1,k+1],
                    [i-1,j-1,k],
                    [i-1,j-1,k-1],
                ]
            elif k==0:
                next_cells = [
                    [i,j-1,k+1],
                    [i,j-1,k],
                    [i-1,j-1,k+1],
                    [i-1,j-1,k],
                ]
            else:
                next_cells = [
                    [i,j-1,k],
                    [i,j-1,k-1],
                    [i-1,j-1,k],
                    [i-1,j-1,k-1],
                ]
        return next_cells

    