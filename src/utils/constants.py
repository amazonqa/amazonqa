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

SOS = 'SOS'
EOS = 'EOS'

QA = 'QA'
REVIEWS = 'Reviews'

ASIN = 'asin'

REVIEW_TEXT = 'reviewText'
REVIEW_TIME = 'reviewTime'
REVIEWS_LIST = 'reviewsList'
REVIEW_IDS_LIST = 'reviewIdsList'

QUESTION_TEXT = 'questionText'
QUESTION_TIME = 'questionTime'
QUESTION_TYPE = 'questionType'

QUESTIONS = 'questions'
QUESTIONS_LIST = 'questionsList'
QUESTION_IDS_LIST = 'questionIdsList'

ANSWER_TEXT = 'answerText'
ANSWER_TIME = 'answerTime'
ANSWERS = 'answers'
ANSWER_IDS_LIST = 'answerIdsList'

HELPFUL = 'helpful'
UNHELPFUL = 'unhelpful'
TIME = 'time'
TEXT = 'text'
TYPE = 'type'
IDS = 'ids'

REVIEW_COLUMNS = [
  REVIEW_TEXT,
  HELPFUL,
  REVIEW_TIME
]

QA_DATA_PATH = '../data/answers_multiple'
REVIEWS_DATA_PATH = '../data/reviews_small'
JSON_DATA_PATH = '../data/json_data'
NN_DATA_PATH = '../data/nn'
INPUT_DATA_PATH = '../../data/input'


LENGTH_DICT = [
  {'Category': 'Automotive', 'ReviewLength': 185, 'QuestionLength': 27, 'AnswerLength': 66},
  {'Category': 'Baby', 'ReviewLength': 219, 'QuestionLength': 28, 'AnswerLength': 64},
  {'Category': 'Beauty', 'ReviewLength': 187, 'QuestionLength': 26, 'AnswerLength': 63},
  {'Category': 'Cell_Phones_and_Accessories', 'ReviewLength': 218, 'QuestionLength': 26, 'AnswerLength': 53},
  {'Category': 'Clothing_Shoes_and_Jewelry', 'ReviewLength': 142, 'QuestionLength': 25, 'AnswerLength': 53},
  {'Category': 'Electronics', 'ReviewLength': 269, 'QuestionLength': 28, 'AnswerLength': 70},
  {'Category': 'Grocery_and_Gourmet_Food', 'ReviewLength': 175, 'QuestionLength': 25, 'AnswerLength': 63},
  {'Category': 'Health_and_Personal_Care', 'ReviewLength': 206, 'QuestionLength': 26, 'AnswerLength': 65},
  {'Category': 'Home_and_Kitchen', 'ReviewLength': 222, 'QuestionLength': 26, 'AnswerLength': 62},
  {'Category': 'Musical_Instruments', 'ReviewLength': 195, 'QuestionLength': 27, 'AnswerLength': 72},
  {'Category': 'Office_Products', 'ReviewLength': 397, 'QuestionLength': 28, 'AnswerLength': 68},
  {'Category': 'Patio_Lawn_and_Garden', 'ReviewLength': 335, 'QuestionLength': 28, 'AnswerLength': 74},
  {'Category': 'Pet_Supplies', 'ReviewLength': 202, 'QuestionLength': 28, 'AnswerLength': 78},
  {'Category': 'Sports_and_Outdoors', 'ReviewLength': 190, 'QuestionLength': 26, 'AnswerLength': 64},
  {'Category': 'Tools_and_Home_Improvement', 'ReviewLength': 240, 'QuestionLength': 27, 'AnswerLength': 68},
  {'Category': 'Toys_and_Games', 'ReviewLength': 213, 'QuestionLength': 26, 'AnswerLength': 55},
  {'Category': 'Video_Games', 'ReviewLength': 462, 'QuestionLength': 26, 'AnswerLength': 56}
]