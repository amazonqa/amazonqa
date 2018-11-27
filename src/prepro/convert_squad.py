import argparse
import string
import math
import json

import pandas as pd
import scipy.stats as st
from evaluator.evaluator import COCOEvalCap
from operator import itemgetter, attrgetter
from tqdm import tqdm


def find_answer_spans(args, answer_span_lens, answers, context):
	context = context.split()

	gold_answers_dict = {}
	gold_answers_dict[0] = [answer["answerText"] for answer in answers]
	answers = []
	
	for answer_span_len in answer_span_lens:
		char_index = 0
		for word_index in range(len(context)-answer_span_len):
			span = ' '.join(context[word_index: word_index+answer_span_len])
			
			generated_answer_dict = {}
			generated_answer_dict[0] = [span]
			
			score = COCOEvalCap.compute_scores(gold_answers_dict, generated_answer_dict)[args.evaluation_metric]
			answers.append((score, {
				'answer_start': char_index,
				'text': span    
			}))
			char_index += (len(context[word_index]) + 1)

	return [i[1] for i in sorted(answers, reverse=True, key=itemgetter(0))[:args.span_max_num]]


def main(args):
	answer_span_lens = range(args.span_min_len, args.span_max_len)
	
	wfp = open(args.output_file, 'w')
	rfp = open(args.input_file, 'r')
	qid = 0

	for line in tqdm(rfp):
		row = json.loads(line)

		#if row["questionType"] == "yesno":
		#	continue

		reviews = row["review_snippets"]
		context = ' '.join(' '.join(reviews).split())

		answers = row["answers"]
		new_answers = find_answer_spans(args, answer_span_lens, answers, context)

		qas = []
		qas.append({
			'id': qid,
			'is_impossible': False,
			'question': row["questionText"],
			'answers': new_answers,
			'human_answers': [answer["answerText"] for answer in answers],
		})

		wfp.write(json.dumps({
				'context': context,
				'qas': qas,
		}) + '\n')

		qid += 1

	wfp.close()


if __name__ == '__main__':
	# parse arguments
	argParser = argparse.ArgumentParser(description="Convert Amazon QAR to Squad format")
	argParser.add_argument("--input_file", type=str)
	argParser.add_argument("--output_file", type=str)
	argParser.add_argument("--span_min_len", type=int, default=2)
	argParser.add_argument("--span_max_len", type=int, default=10)
	argParser.add_argument("--span_max_num", type=int, default=5)
	argParser.add_argument("--evaluation_metric", type=str, default="Bleu_2")

	args = argParser.parse_args()
	main(args)

