import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import json
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


def process_file(config, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    
    if data_type == "dev":
        rfp = open(config.dev_file, 'r')
        wfp = open(config.dev_examples_file, 'w')
        wfp_eval = open(config.dev_eval_file, 'w')
    elif data_type == "test":
        rfp = open(config.test_file, 'r')
        wfp = open(config.test_examples_file, 'w')
        wfp_eval = open(config.test_eval_file, 'w')
    elif data_type == "train":
        rfp = open(config.train_file, 'r')
        wfp = open(config.train_examples_file, 'w')
        wfp_eval = open(config.train_eval_file, 'w')
    else:
        exit(1)

    total = 0

    for line in tqdm(rfp):
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
            wfp.write(json.dumps(example) + '\n')

            row = {"context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"], "id":total}
            wfp_eval.write(json.dumps(row) + '\n')

    rfp.close()
    wfp_eval.close()
    wfp.close()
    #shuffle wfp
    return


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None, token2idx_dict=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit][0:50000]

    assert vec_size is not None
    for token in filtered_elements:
        embedding_dict[token] = [np.random.normal(scale=0.01) for _ in range(vec_size)]
    print("{} tokens have corresponding embedding vector".format(len(filtered_elements)))

    if emb_file is not None:
        assert size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in filtered_elements:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 2)} if token2idx_dict is None else token2idx_dict
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]

    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(config, data_type, word2idx_dict, char2idx_dict, is_test=False):

    if data_type == "dev":
        out_file = config.dev_record_file
        rfp = open(config.dev_examples_file, 'r')
    elif data_type == "test":
        out_file = config.test_record_file
        rfp = open(config.test_examples_file, 'r')
    elif data_type == "train":
        out_file = config.train_record_file
        rfp = open(config.train_examples_file, 'r')
    else:
        exit(1)

    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    char_limit = config.char_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for line in tqdm(rfp):
        example = json.loads(line)
        total_ += 1

        if len(example["context_tokens"]) > para_limit:
            example["context_tokens"] = example["context_tokens"][0:para_limit]
            example["context_chars"] = example["context_chars"][0:para_limit]

        if len(example["ques_tokens"]) > ques_limit:
            example["ques_tokens"] = example["ques_tokens"][0:ques_limit]
            example["ques_chars"] = example["ques_chars"][0:ques_limit]

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example["y1s"][-1], example["y2s"][-1]
        if start < len(y1) and end < len(y2):
            y1[start], y2[end] = 1.0, 1.0

        record = tf.train.Example(features=tf.train.Features(feature={
                                  "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                                  "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                                  "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                                  "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                                  "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                                  "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                                  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
                                  }))
        writer.write(record.SerializeToString())
    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    rfp.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    with open(filename, "w") as fh:
        json.dump(obj, fh)


def load(filename, message=None):
    if message is not None:
        print("Loading {}...".format(message))
    with open(filename, "r") as fh:
        obj = json.load(fh)
    return obj


def save_pickle(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    with open(filename, "wb") as fh:
        pickle.dump(obj, fh)


def load_pickle(filename, message=None):
    if message is not None:
        print("Loading {}...".format(message))
    with open(filename, "rb") as fh:
        obj = pickle.load(fh)
    return obj


def prepro(config):
    # word_counter, char_counter = Counter(), Counter()
    
    # process_file(config, "train", word_counter, char_counter)
    # process_file(config, "dev", word_counter, char_counter)
    # process_file(config, "test", word_counter, char_counter)

    # save_pickle(config.word_counter_file, word_counter, message="word counter")
    # save_pickle(config.char_counter_file, char_counter, message="char counter")

    word_counter = load_pickle(config.word_counter_file)
    char_counter = load_pickle(config.char_counter_file)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word2idx_dict = None
    if os.path.isfile(config.word2idx_file):
        with open(config.word2idx_file, "r") as fh:
            word2idx_dict = json.load(fh)
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file,size=config.glove_word_size, 
        vec_size=config.glove_dim, token2idx_dict=word2idx_dict)

    char2idx_dict = None
    if os.path.isfile(config.char2idx_file):
        with open(config.char2idx_file, "r") as fh:
            char2idx_dict = json.load(fh)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, size=char_emb_size, 
        vec_size=char_emb_dim, token2idx_dict=char2idx_dict)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")

    save(config.word2idx_file, word2idx_dict, message="word2idx")
    save(config.char2idx_file, char2idx_dict, message="char2idx")

    # word_emb_mat = load(config.word_emb_file, message="word embedding")
    # char_emb_mat = load(config.char_emb_file, message="char embedding")

    # word2idx_dict = load(config.word2idx_file, message="word2idx")
    # char2idx_dict = load(config.char2idx_file, message="char2idx")

    train_meta = build_features(config, "train", word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, "dev", word2idx_dict, char2idx_dict)
    test_meta = build_features(config, "test", word2idx_dict, char2idx_dict, is_test=True)

    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")

