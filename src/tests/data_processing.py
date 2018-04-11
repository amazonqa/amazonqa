import os
import argparse
import pandas as pd
from tqdm import tqdm

import constants as C
from language_models.dataloader import AmazonDataLoader
from language_models.dataset import AmazonDataset
from language_models.vocabulary import Vocabulary
from language_models import utils


def _extract_input_attributes(inputs, model_name):
	if model_name == C.LM_ANSWERS:
		answer_seqs, answer_lengths = inputs
		quesion_seqs, review_seqs = None, None
	elif model_name == C.LM_QUESTION_ANSWERS:
		(answer_seqs, answer_lengths), quesion_seqs = inputs
		review_seqs = None
	elif model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
		(answer_seqs, answer_lengths), quesion_seqs, review_seqs = inputs
	else:
		raise 'Unimplemented model: %s' % model_name

	return answer_seqs, quesion_seqs, review_seqs, answer_lengths


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


if __name__ == "__main__":
	# parse arguments
	parser = argparse.ArgumentParser(description="Test AmazonDataset and AmazonDataLoader")
	parser.add_argument("--model_name", type=str, default='LM_A')
	parser.add_argument("--category", type=str, default='Dummy')
	args, _ = parser.parse_known_args()

	model_name = args.model_name
	params = utils.get_model_params(model_name)
	category = params[C.CATEGORY]

	dataset = AmazonDataset(category, model_name, params)
	answersDict, questionsDict, reviewsDict, data = dataset.train

	train_loader = AmazonDataLoader(dataset.train, model_name, params[C.BATCH_SIZE])
	print_dataframe(category, 'train')

	for batch_itr, inputs in enumerate(tqdm(train_loader)):
		answer_seqs, question_seqs, review_seqs, answer_lengths = _extract_input_attributes(inputs, model_name)
		print("batch number ", batch_itr)
		assert(len(answer_seqs) == len(question_seqs) == len(review_seqs))
		
		for i in range(len(answer_seqs)):
			print("batch seq number ", i)
			print(" ".join(dataset.vocab.token_list_from_indices(answer_seqs[i])))
			assert(len(answer_seqs[i]) == answer_lengths[i])

			if model_name == C.LM_QUESTION_ANSWERS or model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
				print(" ".join(dataset.vocab.token_list_from_indices(question_seqs[i])))
			
			if model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
				for review_seq in review_seqs[i]:
					print(" ".join(dataset.vocab.token_list_from_indices(review_seq)))

