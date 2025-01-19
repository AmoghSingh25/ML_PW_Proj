GROUND_COLOR = [34, 139, 34]  # Forest Green
SAND_COLOR = [244, 164, 96]   # Sandy Brown
WATER_COLOR = [30, 144, 255]  # Dodger Blue
FRAC_GROUND, FRAC_SAND = 0.3, 0.3
DIM_MAP = 100

WAVE_FREQ = 0.2079
WAVE_SPEED = 4.088
WAVE_HEIGHT = 0.564
WAVE_AMPLITUDE = 0.282

WAVE_DECAY = 0.038
WAVE_CUTOFF = 0.1297
WAVE_RETREAT_COEFF = 1.8

WAVE_VOL = 10.1866
WAVE_SPREAD = 12

SAND_PULL = 0.0379
GROUND_PULL = 0.0001

WATER_DECAY = 0.99

# Obstacle (barriers) parameters
OBSTACLE_COORDS = [(DIM_MAP // 2, DIM_MAP // 2)]

# Reverse colors for OpenCV
GROUND_COLOR.reverse()
SAND_COLOR.reverse()
WATER_COLOR.reverse()

PARAM_RANGES = {
    "obstacle_coords": [(0, DIM_MAP), (0, DIM_MAP)],  # Obstacle positions
}