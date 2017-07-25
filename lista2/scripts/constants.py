# -*- coding: utf-8 -*-
import os

SEED = 2017
MAX_FLOAT = float('inf')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Database URLs
URL_CAR_DATABASE = 'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
URL_BALANCE_DATABASE = 'http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
URL_SPIRAL_BATABASE = 'http://cs.joensuu.fi/sipu/datasets/spiral.txt'
URL_JAIN_DATABASE = 'http://cs.joensuu.fi/sipu/datasets/jain.txt'
URL_SERVO_DATABASE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data'

# Data sets filenames
FILENAME_CAR_DATABASE = 'car.data'
FILENAME_BALANCE_DATABASE = 'balance-scale.data'
FILENAME_SPIRAL_BATABASE = 'spiral.txt'
FILENAME_JAIN_DATABASE = 'jain.txt'
FILENAME_SERVO_DATABASE = 'servo.data'
