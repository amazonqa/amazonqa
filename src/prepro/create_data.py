import argparse
import string
import math
import json
import numpy as np
import numpy.random as random

import pandas as pd
import scipy.stats as st
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

import retrieval_models
import classify_question
from classify_question import MeanEmbeddingVectorizer

np.random.seed(0)


def top_reviews_and_scores(question_tokens, review_tokens, inverted_index, reviews, review_ids, select_mode, num_reviews):
    if select_mode == "random":
        scores = list(random.uniform(size=len(reviews)))
    elif select_mode in ["bm25", "indri"]:
        scores = retrieval_models.retrieval_model_scores(question_tokens, review_tokens, inverted_index, select_mode)
    elif select_mode == 'wilson':
        scores = []
        for r in range(len(reviews)):
            counts = reviews[r]['helpful']
            helpful_count = int(counts[0])
            unhelpful_count = int(counts[1]) - helpful_count
            scores.append(_wilson_score(helpful_count, unhelpful_count))
    elif select_mode == 'helpful':
        scores = [r['helpful'] for r in reviews]
    else:
        raise 'Unimplemented Review Select Mode'

    scores, top_review_ids = zip(*sorted(list(zip(scores, review_ids)), reverse=True)) if len(scores) > 0 else ([], [])
    return scores[:num_reviews], top_review_ids[:num_reviews]


def tokenize(text):
    punctuations = string.punctuation.replace("\'", '')

    for ch in punctuations:
        text = text.replace(ch, " " + ch + " ")

    tokens = text.split()
    for i, token in enumerate(tokens):
        if not token.isupper():
            tokens[i] = token.lower()
    return tokens


def process_reviews(reviews, review_max_len, stop_words):
    review_tokens = []
    review_texts = []
    all_sentences = []
    for review in reviews:
        sentences = nltk.sent_tokenize(review["reviewText"])
        all_sentences += sentences
        bufr = []
        buffer_len = 0
        for sentence in sentences:
            buffer_len += len(tokenize(sentence))
            bufr.append(sentence)
            if buffer_len > review_max_len:
                review_texts.append(' '.join(bufr))
                bufr = []
                buffer_len = 0

        if buffer_len > 0:
            review_texts.append(' '.join(bufr))

    def get_tokens(texts):
        text_tokens = [tokenize(r) for r in texts]
        return [[token for token in r if token not in stop_words and token not in string.punctuation] for r in text_tokens]

    review_tokens = get_tokens(review_texts)
    sentence_tokens = get_tokens(all_sentences)
    return review_texts, review_tokens, all_sentences, sentence_tokens


def create_inverted_index(review_tokens):
    term_dict = {}
    # TODO: Use actual review IDs
    for doc_id, tokens in enumerate(review_tokens):
        for token in tokens:
            if token in term_dict:
                if doc_id in term_dict[token]:
                    term_dict[token][doc_id] += 1
                else:
                    term_dict[token][doc_id] = 1
            else:
                term_dict[token] = {doc_id: 1}
    return term_dict


def main(args):
    stop_words = set(stopwords.words('english'))
    answer_span_lens = range(2, 10)

    wfp = open(args.output_file, 'w')
    rfp = open(args.input_file, 'r')

    classifier_model_answerable = classify_question.load_classification_model('../../data/model_answerable.pkl')
    #classifier_model_suggestive = classify_question.load_classification_model('../../data/model_suggestive.pkl')
    classifier_vectorizers = classify_question.load_vectorizers('../../data/tfidf_vectorizer.pkl', '../../data/w2v_vectorizer.pkl')

    for line in tqdm(rfp):
        row = json.loads(line)
        if "reviews" not in row:
            print("Wrong Format" + row)
            exit(0)

        reviews = row["reviews"]
        if len(reviews) == 0:
            print("Zero Reviews", row)
            continue

        review_texts, review_tokens, all_sentences, sentence_tokens = process_reviews(reviews, args.review_max_len, stop_words)

        inverted_index = create_inverted_index(review_tokens)
        review_tokens = list(map(set, review_tokens))
        sentence_tokens = list(map(set, sentence_tokens))

        for question in row["questions"]:
            question_text = question["questionText"]
            question_tokens = tokenize(question_text)

            scores_q, top_reviews_q = top_reviews_and_scores(
                set(question_tokens),
                review_tokens,
                inverted_index,
                None,
                review_texts,
                args.review_select_mode,
                args.review_select_num
            )

            _, top_reviews_helpful = top_reviews_and_scores(None, None, None, reviews, [review["reviewText"] for review in reviews], 'helpful', 1)
            _, top_reviews_wilson = top_reviews_and_scores(None, None, None, reviews, [review["reviewText"] for review in reviews], 'wilson', 1)

            _, top_sentences_ir = top_reviews_and_scores(
                set(question_tokens),
                sentence_tokens,
                inverted_index,
                all_sentences,
                all_sentences,
                args.review_select_mode,
                args.review_select_num
            )
            _, top_sentences_random = top_reviews_and_scores(None, None, None, all_sentences, all_sentences, 'random', 1)

            if len(top_reviews_q) == 0:
                print("Zero Top Reviews", row)
                continue

            final_json = {}
            final_json['asin'] = row['asin']
            final_json['category'] = row['category']
            final_json['questionText'] = question_text
            final_json['questionType'] = question["questionType"]
            final_json['review_snippets'] = top_reviews_q
            final_json['random_sentence'] = top_sentences_random
            final_json['top_sentences_IR'] = top_sentences_ir
            final_json['top_review_wilson'] = top_reviews_wilson
            final_json['top_review_helpful'] = top_reviews_helpful
            final_json['answers'] = question["answers"]
            final_json['is_answerable'] = classify_question.is_answerable(classifier_model_answerable, classifier_vectorizers, question_text, top_reviews_q)
            #final_json['is_suggestive'] = 0 if (final_json['is_answerable'] == 0) else classify_question.is_suggestive(classifier_model_suggestive, classifier_vectorizers, question_text, top_reviews_q)
            #print(final_json)
            wfp.write(json.dumps(final_json) + '\n')
    wfp.close()

def _wilson_score(positive, negative):
    confidence = 0.98
    n = positive + negative
    if n == 0:
        return 0.0
    phat = (1.0 * positive) / n
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    return (phat + z*z/(2*n) - z * math.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)


if __name__ == '__main__':
    # parse arguments
    argParser = argparse.ArgumentParser(description="Preprocess QA and Review Data")
    argParser.add_argument("--input_file", type=str)
    argParser.add_argument("--output_file", type=str)
    argParser.add_argument("--review_select_mode", type=str)
    argParser.add_argument("--review_select_num", type=int)
    argParser.add_argument("--review_max_len", type=int, default=100)

    args = argParser.parse_args()
    main(args)
