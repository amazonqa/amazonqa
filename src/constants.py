import torch

# Global constants
USE_CUDA = torch.cuda.is_available()

# Logs related names
LOG_FILENAME = 'logfile'
LOG_DIR = 'log_files'

# Types of RNN Cells
RNN_CELL_LSTM = 'lstm'
RNN_CELL_GRU = 'gru'

# Types of languages models
LM_ANSWERS = 'LM_A'
LM_QUESTION_ANSWERS = 'LM_QA'
LM_QUESTION_ANSWERS_REVIEWS = 'LM_QAR'

RESERVED_IDS = 4

# Special tokens
PAD_TOKEN = '\'PAD\''
EOS_TOKEN = '\'EOS\''
SOS_TOKEN = '\'SOS\''
UNK_TOKEN = '\'UNK\''

PAD_INDEX = 0
EOS_INDEX = 1
SOS_INDEX = 2
UNK_INDEX = 3

# Paths
QA_DATA_PATH = '../data/answers_multiple'
REVIEWS_DATA_PATH = '../data/reviews_small'
JSON_DATA_PATH = '../data/json_data'
INPUT_DATA_PATH = '../data/input'

# Types of categories
AUTOMOTIVE = 'Automotive'
BABY = 'Baby'
BEAUTY = 'Beauty'
CELL_PHONES_AND_ACCESSORIES = 'Cell_Phones_and_Accessories'
CLOTHING_SHOES_AND_JEWELRY = 'Clothing_Shoes_and_Jewelry'
ELECTRONICS = 'Electronics'
GROCERY_AND_GOURMET_FOOD = 'Grocery_and_Gourmet_Food'
HEALTH_AND_PERSONAL_CARE = 'Health_and_Personal_Care'
HOME_AND_KITCHEN = 'Home_and_Kitchen'
MUSICAL_INSTRUMENTS = 'Musical_Instruments'
OFFICE_PRODUCTS = 'Office_Products'
PATIO_LAWN_AND_GARDEN = 'Patio_Lawn_and_Garden'
PET_SUPPLIES = 'Pet_Supplies'
SPORTS_AND_OUTDOORS = 'Sports_and_Outdoors'
TOOLS_AND_HOME_IMPROVEMENT = 'Tools_and_Home_Improvement'
TOYS_AND_GAMES = 'Toys_and_Games'
VIDEO_GAMES = 'Video_Games'

# All Categories
CATEGORY = 'category'
CATEGORIES = [
  AUTOMOTIVE,
  BABY,
  BEAUTY,
  CELL_PHONES_AND_ACCESSORIES,
  CLOTHING_SHOES_AND_JEWELRY,
  ELECTRONICS,
  GROCERY_AND_GOURMET_FOOD,
  HEALTH_AND_PERSONAL_CARE,
  HOME_AND_KITCHEN,
  MUSICAL_INSTRUMENTS,
  OFFICE_PRODUCTS,
  PATIO_LAWN_AND_GARDEN,
  PET_SUPPLIES,
  SPORTS_AND_OUTDOORS,
  TOOLS_AND_HOME_IMPROVEMENT,
  TOYS_AND_GAMES,
  VIDEO_GAMES,
]

QA = 'QA'
REVIEWS = 'Reviews'

ASIN = 'asin'

REVIEW_TEXT = 'reviewText'
REVIEW_TIME = 'reviewTime'
REVIEWS_LIST = 'reviewsList'

QUESTION_TEXT = 'questionText'
QUESTION_TIME = 'questionTime'
QUESTION_TYPE = 'questionType'

QUESTIONS = 'questions'
QUESTIONS_LIST = 'questionsList'

ANSWER_TEXT = 'answerText'
ANSWER_TIME = 'answerTime'
ANSWERS = 'answers'

HELPFUL = 'helpful'
UNHELPFUL = 'unhelpful'
TIME = 'time'
TEXT = 'text'
TYPE = 'type'

"""
TODO: remove if not being used anywhere
REVIEW_COLUMNS = [
  REVIEW_TEXT,
  HELPFUL,
  REVIEW_TIME
]
"""
"""
    Modes
"""
TRAIN_TYPE = 'train'
TEST_TYPE = 'test'
DEV_TYPE = 'val'
"""
Default Hyperparameters for LM MODELS
"""

# Hyperparameter constants
EPOCHS = 'epochs'
BATCH_SIZE = 'batch_size'
DROPOUT = 'dropout'
LR = 'lr'
HDIM_A = 'hdim_a'
HDIM_Q = 'hdim_q'
HDIM_R = 'hdim_r'
EMBEDDING_DIM = 'embedding_dim'
H_LAYERS = 'h_layers'
DECAY_START_EPOCH = 'decay_start_epoch'
LR_DECAY = 'lr_decay'
GLOBAL_NORM_MAX = 'global_norm_max'
VOCAB_SIZE = 'vocab_size'

MAX_QUESTION_LEN = 'max_question_len'
MAX_ANSWER_LEN = 'max_answer_len'
MAX_REVIEW_LEN = 'max_review_len'

TEACHER_FORCING_RATIO = 'teacher_forcing_ratio'
OUTPUT_MAX_LEN = 'output_max_len'
MODEL_NAME = 'model_name'

REVIEW_SELECT_MODE = 'review_select_mode'
REVIEW_SELECT_NUM = 'review_select_num'
"""
    Review select modes
"""
RANDOM = 'random'
HELPFUL = 'helpful'
WILSON = 'wilson'

RESUME = 'resume'
RESUME_LR = 'resume_lr'
SAVE_DIR = 'save_dir'

# DIRS/FILENAMES
BASE_PATH = 'saved_models'
SAVED_MODEL_FILENAME = 'model.pt'
SAVED_PARAMS_FILENAME = 'params.json'
SAVED_VOCAB_FILENAME = 'vocab.pkl'
SAVED_ARCHITECTURE_FILENAME = 'architecture.txt'
SAVED_OPTIMIZER_FILENAME = 'optimizer'

LM_MODELS =                       [LM_ANSWERS,        LM_QUESTION_ANSWERS, LM_QUESTION_ANSWERS_REVIEWS]
"""
    WARNING : LM_HP shouldn't be accessed anywhere other than utils.get_model_params()
"""
LM_HP = {
    MODEL_NAME:                   LM_MODELS,
    REVIEW_SELECT_MODE:           [None,              None,               HELPFUL],
    REVIEW_SELECT_NUM:            [None,              None,               5],
    EPOCHS:                       [25,                25,                 25],
    BATCH_SIZE:                   [64,                64,                 64],
    DROPOUT:                      [0.2,               0.2,                0.2],
    LR:                           [0.01,              0.01,               0.01],
    HDIM_A:                       [512,               512,                512],
    HDIM_Q:                       [None,              512,                256],
    HDIM_R:                       [None,              None,               256],
    EMBEDDING_DIM:                [512,               512,                512],
    H_LAYERS:                     [2,                 2,                  2],
    #DECAY_START_EPOCH:            [10,                10,                 10],
    LR_DECAY:                     [0.5,               0.5,                0.5],
    GLOBAL_NORM_MAX:              [5,                 5,                  5],
    VOCAB_SIZE:                   [20000,             20000,              20000],
    TEACHER_FORCING_RATIO:        [0.9,               0.9,                0.9],
    OUTPUT_MAX_LEN:               [128,               128,                128],
    CATEGORY:                     [VIDEO_GAMES] * 3,
    LOG_FILENAME:                 ['log.log'] * 3,
    MAX_QUESTION_LEN:             [100] * 3,
    MAX_ANSWER_LEN:               [100] * 3,
    MAX_REVIEW_LEN:               [200] * 3,
    RESUME_LR:                    [None]*3, # if resume_lr is none, use optimizers lr  
    SAVE_DIR:                     [None]*3
}
