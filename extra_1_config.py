IMAGE_SIZE = 28
INPUT_LAYER_SIZE = IMAGE_SIZE * IMAGE_SIZE
OUTPUT_LAYER_SIZE = 10

# hidden_layer_size = 500
hidden_layer_number = 7
MODEL_FOLDER_NAME = './models/pa2-{}'
DROPOUT_RATE = 0.5
learning_rate = 1e-4

NOISE_STD = 0.06
TRAINING_DATA_NUMBER = 160000
BATCH_SIZE = 1000
assert TRAINING_DATA_NUMBER % BATCH_SIZE == 0
BATCH_ITER_NUMBER = TRAINING_DATA_NUMBER // BATCH_SIZE
EPOCH = 100
TRAIN_ITER_NUMBER = EPOCH * BATCH_ITER_NUMBER

EVERY_N_ITER = 1000
EARLY_STOP_TRAIN_LOSS = 0.001
EARLY_STOP_VALID_LOSS_MULTIPLIER = 1.5

BN_MOMENTUM = 0.9
