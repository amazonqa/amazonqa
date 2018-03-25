import torch

# Global constants
USE_CUDA = torch.cuda.is_available()

# Types of RNN Cells
RNN_CELL_LSTM = 'lstm'
RNN_CELL_GRU = 'gru'

# Types of languages models
LM_ANSWERS = 'LM_A'
LM_QUESTION_ANSWERS = 'LM_QA'
LM_QUESTION_ANSWERS_REVIEWS = 'LM_QAR'

# Special tokens
PAD_TOKEN = '<PAD>'
EOS_TOKEN = '<EOS>'
SOS_TOKEN = '<SOS>'
UNK_TOKEN = '<UNK>'

PAD_INDEX = 0
EOS_INDEX = 1
SOS_INDEX = 2
UNK_INDEX = 3

# Paths
QA_DATA_PATH = 'data/answers_multiple'
REVIEWS_DATA_PATH = 'data/reviews_small'
JSON_DATA_PATH = 'data/json_data'
NN_DATA_PATH = 'data/nn'
INPUT_DATA_PATH = 'data/input'

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

REVIEW_COLUMNS = [
  REVIEW_TEXT,
  HELPFUL,
  REVIEW_TIME
]
