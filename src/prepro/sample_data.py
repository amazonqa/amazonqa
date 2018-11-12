import argparse
import os
import string
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

MTURK=True

#python3 sample_data.py --input_file ../../data/train-qar_all.jsonl --output_file ../../data/train-qar_sample_100000.csv --num_entries 100000

def getDF(path):
	i = 0
	df = {}
	with open(path, 'r') as fp:
		for line in fp:
			df[i] = json.loads(line)
			i += 1
	
	return pd.DataFrame.from_dict(df, orient='index')


def main(args):
	df = getDF(args.input_file)
	all_questions = len(df)
	num_entries = args.num_entries

	print('Total Questions: %d' % all_questions)
	print('Sampling %d instances ...' % num_entries)
	sample_question_ids = np.random.permutation(all_questions)[:num_entries]

	samples = []
	for qid in tqdm(sample_question_ids):
		row = df.iloc[qid]
		question_text = row["questionText"]
		answers = row["answers"]
		answer_texts = [answer["answerText"] for answer in answers][:3]
		top_reviews_q = row["review_snippets"]
		
		sample_dict = {}
		sample_dict['id'] = '%d' % qid
		sample_dict['category'] = row['category']
		sample_dict['question'] = question_text
		sample_dict['review0'] = get_review(top_reviews_q, 1)
		sample_dict['review1'] = get_review(top_reviews_q, 2)
		sample_dict['review2'] = get_review(top_reviews_q, 3)
		sample_dict['review3'] = get_review(top_reviews_q, 4)
		sample_dict['review4'] = get_review(top_reviews_q, 5)

		samples.append(sample_dict)

	if samples:
		pd.DataFrame(samples).to_csv(args.output_file, index=False)


def get_review(top_reviews_q, i):
	if len(top_reviews_q) < i:
		return ''
	return top_reviews_q[i-1]


if __name__ == '__main__':
	# parse arguments
	argParser = argparse.ArgumentParser(description="Sample Data for MTURK")
	argParser.add_argument("--input_file", type=str)
	argParser.add_argument("--output_file", type=str)
	argParser.add_argument("--num_entries", type=int)
	argParser.add_argument("--seed", type=int, default=1)

	args = argParser.parse_args()
	main(args)


