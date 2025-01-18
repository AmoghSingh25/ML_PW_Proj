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

SAND_PULL = 0.2
GROUND_PULL = 0.1
WATER_DECAY = 0.99

# Reverse colors for OpenCV
GROUND_COLOR.reverse()
SAND_COLOR.reverse()
WATER_COLOR.reverse()

# Parameter ranges for optimization - to be changed for barrier only params
PARAM_RANGES = {
    "wave_freq": (10, 50),
    "wave_speed": (0.5, 2.0),
    "wave_decay": (0.4, 1.0),
    "wave_cutoff": (0.05, 0.2),
    "wave_retreat_coeff": (0.5, 1.0),
    "wave_height": (1, 5),
    "sand_pull": (0.1, 0.5),
    "ground_pull": (0.05, 0.2),
    "water_decay": (0.95, 1.0),
    "wave_vol": (1, 5),
    "wave_amplitude": (1, 5),
    "wave_spread": (0, 5)
}