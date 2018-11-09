import pickle
import pandas as pd
import gzip
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import constants as C
from tqdm import tqdm
import json

np.random.seed(2018)

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def filepath(category, key):
    if key == C.QA:
        path = '%s/QA_%s.json.gz' % (C.QA_DATA_PATH, category)
    elif key == C.REVIEWS:
        path = '%s/reviews_%s_5.json.gz' % (C.REVIEWS_DATA_PATH, category)
    else:
        raise 'Unexpected key'
    return path


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# convert reviews row to json
def reviewToJson(row):
  json = {}

  text = row[C.REVIEW_TEXT]
  matchIdx = text.find("\n\n\n\n")
  json[C.TEXT] = text[0:matchIdx] if matchIdx != -1 else text

  scores = str(row[C.HELPFUL])[1:-1].split(',')
  json[C.HELPFUL] = int(scores[0].strip())
  json[C.UNHELPFUL] = int(scores[1].strip())

  json[C.TIME] = row[C.REVIEW_TIME]
  return json


def questionsToJson(questions_list):
  new_questions_list = []
  for question in questions_list:
    new_question = {}

    text = question[C.QUESTION_TEXT]
    matchIdx = text.find("\n\n\n\n")

    new_question[C.TEXT] = text[0:matchIdx] if matchIdx != -1 else text
    new_question[C.TIME] = question[C.QUESTION_TIME]
    new_question[C.TYPE] = question[C.QUESTION_TYPE]

    new_answers = []
    for answer in question[C.ANSWERS]:
      new_answer = {}

      text = answer[C.ANSWER_TEXT]
      matchIdx = text.find("\n\n\n\n")

      new_answer[C.TEXT] = text[0:matchIdx] if matchIdx != -1 else text
      new_answer[C.TIME] = answer[C.ANSWER_TIME]

      scores = str(answer[C.HELPFUL])[1:-1].split(',')
      new_answer[C.HELPFUL] = int(scores[0].strip())
      new_answer[C.UNHELPFUL] = int(scores[1].strip())
      new_answers.append(new_answer)

    new_question[C.ANSWERS] = new_answers
    new_questions_list.append(new_question)

  return new_questions_list

def generate_raw_data(category):
  qa, reviews = [getDF(filepath(category, key)) for key in [C.QA, C.REVIEWS]]

  reviews[C.REVIEWS_LIST] = reviews[C.REVIEW_COLUMNS].apply(reviewToJson, axis=1)
  reviews = reviews.groupby(C.ASIN).apply(lambda x: x[C.REVIEWS_LIST].tolist()).reset_index()
  reviews.columns = [C.ASIN, C.REVIEWS_LIST]

  qa[C.QUESTIONS_LIST] = qa[C.QUESTIONS].apply(questionsToJson)
  qa = qa[[C.ASIN, C.QUESTIONS_LIST]]

  qa_reviews = pd.merge(qa, reviews, on=[C.ASIN, C.ASIN])

  with open('%s/%s.pickle' % (C.JSON_DATA_PATH, category), 'wb') as f:
    qa_reviews.to_pickle(f)


def generate_raw_data_all_categories():
  for category in tqdm(C.CATEGORIES):
    generate_raw_data(category)


def get_raw_dataframe(category):
    with open('%s/%s.pickle' % (C.JSON_DATA_PATH, category), 'rb') as f:
        return pd.read_pickle(f)


def generate_split_data(category):
  pd = get_raw_dataframe(category)
  length = len(pd)

  indexes = np.array(range(0, length))
  np.random.shuffle(indexes)

  train = (int)(length*60.0/100)
  val = (int)(length*20.0/100)
  test = (int)(length*20.0/100)

  train_pd = pd.iloc[indexes[0 : train]]
  val_pd = pd.iloc[indexes[train : train+val]]
  test_pd = pd.iloc[indexes[train+val : train+val+test]]

  with open('%s/train-%s.pickle' % (C.INPUT_DATA_PATH, category), 'wb') as f:
    train_pd.to_pickle(f)

  with open('%s/val-%s.pickle' % (C.INPUT_DATA_PATH, category), 'wb') as f:
    val_pd.to_pickle(f)

  with open('%s/test-%s.pickle' % (C.INPUT_DATA_PATH, category), 'wb') as f:
    test_pd.to_pickle(f)


def generate_split_data_all_categories():
  for category in C.CATEGORIES:
    generate_split_data(category)


def get_split_dataframe(category, split):
    with open('%s/%s-%s.pickle' % (C.INPUT_DATA_PATH, split, category), 'rb') as f:
        return pd.read_pickle(f)


def load_json_data(category):
  filepath = C.JSON_DATA_PATH+"/"+category+".json"
  print(filepath)
  raw_df = pd.read_json(filepath, orient='index')
  
  with open('%s/%s.pickle' % (C.JSON_DATA_PATH, category), 'wb') as f:
    raw_df.to_pickle(f)


def load_text_data(category, split):
  text_filepath = "%s/%s-%s.txt" % (C.TEXT_DATA_PATH, split, category)
  line_num = 0
  full_json = {}

  with open(text_filepath, 'r') as f:
    for line in f:
      answer_json = {C.TEXT: line.strip()}

      question_json = {C.ANSWERS: [answer_json], C.TEXT: ""}

      product_json = {C.ASIN: line_num, C.QUESTIONS_LIST: [question_json], C.REVIEWS_LIST: []}

      full_json[line_num] = product_json
      line_num += 1


  df = pd.DataFrame.from_dict(full_json, orient='index')
  pickle_filepath = "%s/%s-%s.pickle" % (C.INPUT_DATA_PATH, split, category)

  with open(pickle_filepath, 'wb') as f:
    df.to_pickle(f)


def load_ms_marco_data(split):
  category = "Ms_Marco"
  json_filepath = "%s/%s-%s.jsonl" % (C.JSON_DATA_PATH, split, category)
  full_json = {}
  line_num = 0

  with open(json_filepath, 'r') as f:
      for line in f:
          j = json.loads(line)
          
          answers = j["answers"]
          num_answers = len(answers)
          
          if num_answers == 1 and answers[0] == 'No Answer Present.':
              continue
          
          answers_list = []
          for answer in answers:
              answer_text = answer.strip()
              answer_json = {C.TEXT: answer_text}
              answers_list.append(answer_json)
          
          question_text = j["query"].strip()
          question_json = {C.ANSWERS: answers_list, C.TEXT: question_text}
          
          reviews_list = []
          for passage in j["passages"]:
              review_text = passage["passage_text"].strip()
              review_json = {C.TEXT: review_text}
              reviews_list.append(review_json)
          
          product_json = {C.ASIN: line_num, C.QUESTIONS_LIST: [question_json], C.REVIEWS_LIST: reviews_list}

          full_json[line_num] = product_json
          line_num += 1
      
  df = pd.DataFrame.from_dict(full_json, orient='index')
  pickle_filepath = "%s/%s-%s.pickle" % (C.INPUT_DATA_PATH, split, category)

  with open(pickle_filepath, 'wb') as f:
    df.to_pickle(f)

