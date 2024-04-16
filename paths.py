import os

DATA_PATH = "./tiny-imagenet-200"
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
VALID_PATH = os.path.join(DATA_PATH, 'val') 
VALID_PATH_2 = VALID_PATH + "/images2"
TEST_PATH = os.path.join(DATA_PATH, 'test')