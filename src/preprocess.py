
import pickle
import pandas as pd
import gzip
import numpy as np
import nltk
from nltk.corpus import stopwords
import random
import string
import constants as C
import data_utils as D


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
  qa, reviews = [D.getDF(D.filepath(category, key)) for key in [C.QA, C.REVIEWS]]

  reviews[C.REVIEWS_LIST] = reviews[C.REVIEW_COLUMNS].apply(reviewToJson, axis=1)
  reviews = reviews.groupby(C.ASIN).apply(lambda x: x[C.REVIEWS_LIST].tolist()).reset_index()
  reviews.columns = [C.ASIN, C.REVIEWS_LIST]

  qa[C.QUESTIONS_LIST] = qa[C.QUESTIONS].apply(questionsToJson)
  qa = qa[[C.ASIN, C.QUESTIONS_LIST]]

  qa_reviews = pd.merge(qa, reviews, on=[C.ASIN, C.ASIN])
  #qa_reviews_json = qa_reviews.to_json()

  with open('%s/%s.pickle' % (C.JSON_DATA_PATH, category), 'wb') as f:
    qa_reviews.to_pickle(f)


def generate_raw_data_all_categories():
  for category in CATEGORIES:
    generate_raw_data(category)


def raw_dataframe(category):
    with open('%s/%s.pickle' % (C.JSON_DATA_PATH, category), 'rb') as f:
        return pd.read_pickle(f)


