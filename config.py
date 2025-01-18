GROUND_COLOR = [34, 139, 34]  # Forest Green
SAND_COLOR = [244, 164, 96]   # Sandy Brown
WATER_COLOR = [30, 144, 255]  # Dodger Blue
FRAC_GROUND, FRAC_SAND = 0.3, 0.3
DIM_MAP = 100

WAVE_FREQ = 20
WAVE_SPEED = 5.344
WAVE_DECAY = 0.6
WAVE_CUTOFF = 0.1
WAVE_RETREAT_COEFF = 0.8
WAVE_HEIGHT = 0.572
WAVE_VOL = 1
WAVE_AMPLITUDE = 50
WAVE_SPREAD = 0

SAND_PULL = 0.2
GROUND_PULL = 0.1
WATER_DECAY = 0.99

# Obstacle (barriers) parameters
OBSTACLE_COORDS = [(DIM_MAP // 2, DIM_MAP // 2)]

# Reverse colors for OpenCV
GROUND_COLOR.reverse()
SAND_COLOR.reverse()
WATER_COLOR.reverse()

# Parameter ranges for optimization - to be changed for barrier only params
PARAM_RANGES = {
    "obstacle_coords": [(0, DIM_MAP), (0, DIM_MAP)], 
}