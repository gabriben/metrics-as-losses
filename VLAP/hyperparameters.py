# global BATCH_SIZE
# global SHUFFLE_BUFFER_SIZE
# global IMG_SIZE
# global CHANNELS

# class hypers():
BATCH_SIZE = 256 # Big enough to measure an F1-score
SHUFFLE_BUFFER_SIZE = 1024 # Shuffle the training data by a chunck of 1024 observations
IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
CHANNELS = 3 # Keep RGB color channels to match the input format of the model
LR = 1e-5 # Keep it small when transfer learning
EPOCHS = 100
LOSS_FUNCTION = "sigmoidF1"
S = -10 # sigmoid s-shape hyperparam
E = 1 # sigmoid offset hyperparam

METRIC = "macroF1"
MACRO_F1_THRESH = 0.5
LAST_ACTIVATION = "sigmoid"

# TENCENT (original descriptions to the right)

# TENCENT_MASK_THRESH : 0.7 # "mask thres for balance pos neg"
# TENCENT_CLASS_NUM : 1000 # "distinct class number"
# TENCENT_RANDOM_SEED : 1234 #  "Random sedd for neigitive class selected"
# TENCENT_WEIGHT_DECAY : 0.0001 # "Tainable Weight l2 loss factor."
# TENCENT_NEG_SELECT : 0.3 # "how many class within only negtive samples in a batch select to learn"

