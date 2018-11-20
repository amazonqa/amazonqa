import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
import os.path
import pickle

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        for line in tqdm(fh):
            para = json.loads(line)
            context = para["context"].replace(
                "''", '" ').replace("``", '" ')
            context_tokens = word_tokenize(context)
            context_chars = [list(token) for token in context_tokens]
            spans = convert_idx(context, context_tokens)
            for token in context_tokens:
                word_counter[token] += len(para["qas"])
                for char in token:
                    char_counter[char] += len(para["qas"])
            for qa in para["qas"]:
                total += 1
                ques = qa["question"].replace(
                    "''", '" ').replace("``", '" ')
                ques_tokens = word_tokenize(ques)
                ques_chars = [list(token) for token in ques_tokens]
                for token in ques_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1
                y1s, y2s = [], []
                answer_texts = []
                for answer in qa["answers"]:
                    answer_text = answer["text"]
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                           "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                examples.append(example)
                eval_examples[str(total)] = {
                    "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def process_train_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        for line in tqdm(fh):
            para = json.loads(line)
            context = para["context"].replace(
                "''", '" ').replace("``", '" ')
            context_tokens = word_tokenize(context)
            spans = convert_idx(context, context_tokens)
            for token in context_tokens:
                word_counter[token] += len(para["qas"])
                for char in token:
                    char_counter[char] += len(para["qas"])
            for qa in para["qas"]:
                total += 1
                ques = qa["question"].replace(
                    "''", '" ').replace("``", '" ')
                ques_tokens = word_tokenize(ques)
                for token in ques_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1
                answer_texts = []
                for answer in qa["answers"]:
                    answer_text = answer["text"]
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)

                eval_examples[str(total)] = {
                    "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
    return eval_examples


def save_pickle(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    with open(filename, "wb") as fh:
        pickle.dump(obj, fh)


def save_json(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    with open(filename, "w") as fh:
        json.dump(obj, fh)


def prepro(config):
    word_counter, char_counter = Counter(), Counter()

    train_eval = process_train_file(
        config.train_file, "train", word_counter, char_counter)
    save_json(config.train_eval_file, train_eval, message="train eval")

    dev_examples, dev_eval = process_file(
        config.dev_file, "dev", word_counter, char_counter)
    save_json(config.dev_eval_file, dev_eval, message="dev eval")
    save_pickle(config.dev_examples_file, dev_examples, message="dev examples")

    test_examples, test_eval = process_file(
        config.test_file, "test", word_counter, char_counter)
    save_json(config.test_eval_file, test_eval, message="test eval")
    save_pickle(config.test_examples_file, test_examples, message="test examples")

    save_pickle(config.word_counter_file, word_counter, message="word counter")
    save_pickle(config.char_counter_file, char_counter, message="char counter")

