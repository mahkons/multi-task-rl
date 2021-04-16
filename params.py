ENV_NAMES = ["AntBulletEnv-v0", "HalfCheetahBulletEnv-v0", "HumanoidBulletEnv-v0",
        "HopperBulletEnv-v0", "Walker2DBulletEnv-v0", "InvertedPendulumBulletEnv-v0",
        "InvertedDoublePendulumBulletEnv-v0", "InvertedPendulumSwingupBulletEnv-v0"]

LAMBDA = 0.95
GAMMA = 0.99

LR = 2e-4
VALUE_COEFF = 0.5

CLIP = 0.1
ENTROPY_COEF = 1e-2
BATCHES_PER_UPDATE = 32
BATCH_SIZE = 64

MIN_TRANSITIONS_PER_UPDATE = 256
MIN_EPISODES_PER_UPDATE = 1

ITERATIONS = 100
