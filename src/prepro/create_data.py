import argparse
import string
import math
import json

import pandas as pd
import scipy.stats as st
import nltk
from nltk.corpus import stopwords
from operator import itemgetter, attrgetter

from evaluator.evaluator import COCOEvalCap
import retrieval_models


def top_reviews_and_scores(question_tokens, review_tokens, inverted_index, reviews, review_ids, select_mode, num_reviews):
	if select_mode == "random":
		scores = list(random.uniform(size=len(reviews)))
	elif select_mode in ["bm25", "indri"]:
		scores = retrieval_models.retrieval_model_scores(question_tokens, review_tokens, inverted_index, select_mode)
	else:
		raise 'Unimplemented Review Select Mode'
	
	top_review_ids, scores = zip(*sorted(list(zip(scores, review_ids)), reverse=True)) if len(scores) > 0 else ([], [])
	return top_review_ids[:num_reviews], scores[:num_reviews]


def tokenize(text):
	punctuations = string.punctuation.replace("\'", '')

	for ch in punctuations:
		text = text.replace(ch, " " + ch + " ")

	tokens = text.split()
	for i, token in enumerate(tokens):
		if not token.isupper():
			tokens[i] = token.lower()
	return tokens


def process_reviews(reviews, max_review_len, stop_words):
	review_tokens = []
	review_texts = []
	for review in reviews:
		sentences = nltk.sent_tokenize(review["reviewText"])
		bufr = []
		buffer_len = 0
		for sentence in sentences:
			buffer_len += len(tokenize(sentence))
			if buffer_len > max_review_len:
				review_texts.append(' '.join(bufr))
				bufr = []
				buffer_len = 0
			else:
				bufr.append(sentence)

	review_tokens = [tokenize(r) for r in review_texts]
	review_tokens = [[token for token in r if token not in stop_words and token not in string.punctuation] for r in review_tokens]
	return review_texts, review_tokens


def find_answer_spans(max_num_spans, answer_span_lens, answers, context):
	context = context.split(' ')

	gold_answers_dict = {}
	gold_answers_dict[0] = [answer["answerText"] for answer in answers]
	answers = []
	
	for answer_span_len in answer_span_lens:
		char_index = 0
		for word_index in range(len(context)-answer_span_len):
			span = ' '.join(context[word_index: word_index+answer_span_len])
			
			generated_answer_dict = {}
			generated_answer_dict[0] = [span]
			
			score = COCOEvalCap.compute_scores(gold_answers_dict, generated_answer_dict)['Bleu_2']
			answers.append((score, {
				'answer_start': char_index,
				'text': span    
			}))
			char_index += (len(context[word_index]) + 1)

	return [i[1] for i in sorted(answers, reverse=True, key=itemgetter(0))[:max_num_spans]]


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
	df = pd.read_json(args.input_file, orient='records', lines=True, compression='gzip')

	stop_words = set(stopwords.words('english'))
	answer_span_lens = range(2, 10)

	with open(args.output_file, 'w') as fp:
		for (_, row) in df.iterrows():
			reviews = row["reviews"]
			if len(reviews) == 0:
				continue

			review_texts, review_tokens = process_reviews(reviews, args.max_review_len, stop_words)

			inverted_index = create_inverted_index(review_tokens)
			review_tokens = list(map(set, review_tokens))

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
				context = ' '.join(top_reviews_q)

				answers = question["answers"]
				new_answers = find_answer_spans(args.max_num_spans, answer_span_lens, answers, context)

				final_json = {}
				final_json['asin'] = row['asin']
				final_json['category'] = row['category']
				final_json['questionText'] = question_text
				final_json['questionType'] = question["questionType"]
				final_json['review_snippets'] = context
				final_json['answers'] = answers
				final_json['answers_spans'] = new_answers

				fp.write(json.dumps(final_json) + '\n')


if __name__ == '__main__':
	# parse arguments
	argParser = argparse.ArgumentParser(description="Preprocess QA and Review Data")
	argParser.add_argument("--input_file", type=str)
	argParser.add_argument("--output_file", type=str)
	argParser.add_argument("--review_select_mode", type=str)
	argParser.add_argument("--review_select_num", type=int)
	argParser.add_argument("--max_review_len", type=int, default=100)
	argParser.add_argument("--max_num_spans", type=int, default=10)

	args = argParser.parse_args()
	main(args)


