import torch

WORD_LOSS = 'WORD_LOSS'
SENTENCE_LOSS = 'SENTENCE_LOSS'


# Global constants
USE_CUDA = torch.cuda.is_available()

# Types of RNN Cells
RNN_CELL_LSTM = 'lstm'
RNN_CELL_GRU = 'gru'

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
TEXT_DATA_PATH = '../data/text_data'
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
REVIEW_SNIPPETS = 'review_snippets'

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


#TODO: remove if not being used anywhere
REVIEW_COLUMNS = [
  REVIEW_TEXT,
  HELPFUL,
  REVIEW_TIME
]

"""
    Modes
"""
TRAIN_TYPE = 'train'
TEST_TYPE = 'test'
DEV_TYPE = 'dev'
"""
Default Hyperparameters for LM MODELS
"""

# Hyperparameter constants
NUM_EPOCHS = 'num_epochs'
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
USE_ATTENTION = 'use_attention'

MAX_QUESTION_LEN = 'max_question_len'
MAX_ANSWER_LEN = 'max_answer_len'
MAX_REVIEW_LEN = 'max_review_len'

TEACHER_FORCING_RATIO = 'teacher_forcing_ratio'
OUTPUT_MAX_LEN = 'output_max_len'
MODEL_NAME = 'model_name'
OPTIMIZER_TYPE = 'optimizer_type'

REVIEW_SELECT_MODE = 'review_select_mode'
REVIEW_SELECT_NUM = 'review_select_num'

"""
    Review select modes
"""
RANDOM = 'random'
HELPFUL = 'helpful'
WILSON = 'wilson'
BM25 = 'bm25'
INDRI = 'indri'

"""
    Optimizer types
"""
ADAM = 'adam'
SGD = 'sgd'

RESUME = 'resume'
SAVE_DIR = 'save_dir'
BEST_EPOCH_IDX = -100

# DIRS/FILENAMES
BASE_PATH = 'saved_models'

# Logs related names
LOG_FILENAME = 'logfile'
LOG_DIR = '%s/log_files' % BASE_PATH
