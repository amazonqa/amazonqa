import os
import argparse
import pandas as pd
from tqdm import tqdm

from preprocessing import preprocess as P
import constants as C
from data.dataloader import AmazonDataLoader
from data.dataset import AmazonDataset
from data.vocabulary import Vocabulary
import config


def _extract_input_attributes(inputs, model_name):
	if model_name == C.LM_ANSWERS:
		answer_seqs, answer_lengths = inputs
		question_seqs, review_seqs, question_ids = None, None, None
	elif model_name == C.LM_QUESTION_ANSWERS:
		(answer_seqs, answer_lengths), question_seqs, question_ids = inputs
		review_seqs = None
	elif model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
		(answer_seqs, answer_lengths), question_seqs, question_ids, review_seqs = inputs
	else:
		raise 'Unimplemented model: %s' % model_name

	return answer_seqs, question_seqs, question_ids, review_seqs, answer_lengths


def print_dataframe(category, split):
	split = 'train'
	with open('%s/%s-%s.pickle' % (C.INPUT_DATA_PATH, split, category), 'rb') as f:
		dataframe = pd.read_pickle(f)

	for index, row in tqdm(dataframe.iterrows()):
		questionsList = row[C.QUESTIONS_LIST]
		for question in questionsList:
			print(question[C.TEXT])

			for answer in question[C.ANSWERS]:
				print(answer[C.TEXT])

		reviewsList = row[C.REVIEWS_LIST]
		for review in reviewsList:
			print(review[C.TEXT])


def preprocess_data(category):
	P.load_json_data(category)
	P.generate_split_data(category)


if __name__ == "__main__":
	# parse arguments
	parser = argparse.ArgumentParser(description="Test AmazonDataset and AmazonDataLoader")
	parser.add_argument("--model_name", type=str, default='LM_A')
	parser.add_argument("--category", type=str, default='Dummy')
	parser.add_argument("--max_question_len", type=int, default=100)
	parser.add_argument("--max_answer_len", type=int, default=200)
	parser.add_argument("--max_review_len", type=int, default=300)
	args, _ = parser.parse_known_args()

	model_name = args.model_name
	params = config.get_model_params(model_name)
	params[C.CATEGORY]  = args.category

	#preprocess_data(params[C.CATEGORY])

	dataset = AmazonDataset(params)
	answersDict, questionsDict, questionAnswersDict, reviewsDict, data = dataset.test
	print(answersDict)
	print(questionsDict)
	print(questionAnswersDict)

	test_loader = AmazonDataLoader(dataset.test, model_name, params[C.BATCH_SIZE])
	#print_dataframe(params[C.CATEGORY], 'test')

	for batch_itr, inputs in enumerate(tqdm(test_loader)):
		answer_seqs, question_seqs, question_ids, review_seqs, answer_lengths = _extract_input_attributes(inputs, model_name)
		print("batch number ", batch_itr)
		print(question_ids)

		for i in range(len(answer_seqs)):
			print("batch seq number ", i)
			print(answer_seqs[i])
			#print(" ".join(dataset.vocab.token_list_from_indices(answer_seqs[i])))
			#assert(len(answer_seqs[i]) == answer_lengths[i])

			if model_name == C.LM_QUESTION_ANSWERS or model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
				#print(" ".join(dataset.vocab.token_list_from_indices(question_seqs[i])))
				print(question_seqs[i])
			
		if model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
			for i in range(len(review_seqs)):
				for review_seq in review_seqs[i]:
					print(" ".join(dataset.vocab.token_list_from_indices(review_seq)))

