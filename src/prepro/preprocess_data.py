import json
import gzip
import wget
import argparse
from tqdm import tqdm
import pandas as pd


def parse(path):
	g = gzip.open(path, 'r')
	for l in tqdm(g):
		yield eval(l)


def getDF(path):
	i = 0
	df = {}
	for d in parse(path):
		df[i] = d
		i += 1
	return pd.DataFrame.from_dict(df, orient='index')


def clean_text(text):
	matchIdx = text.find("\n\n\n\n")
	text = text[0:matchIdx] if matchIdx != -1 else text
	return text


def get_answer_type(answer):
	answer_type = 'NA'

	if 'answerType' in answer:
		if answer['answerType'] != '?':
			answer_type = answer['answerType']
	return answer_type


def get_question_type(question):
	question_type = ''

	if question['questionType'] == "open-ended":
		question_type = "descriptive"
	elif question['questionType'] == "yes/no":
		question_type = "yesno"
	else:
		print("Unknown QuestionType " + question)
		exit(0)
	return question_type


def download_data(data_dir, category, url_prefix):
	url_suffix = '.json.gz'

	download_link = url_prefix + category + url_suffix
	wget.download(download_link, data_dir)


def clean_questions(questions_list):
	questions = []
	for q in questions_list:
		question = {}
		
		question['questionText'] = clean_text(q['questionText'])
		question['questionType'] = get_question_type(q)

		answers = []
		for a in q['answers']:
			answer = {}

			answer['answerText'] = clean_text(a['answerText'])
			answer['answerType'] = get_answer_type(a)
			answer['helpful'] = a['helpful']
			answers.append(answer)

		question['answers'] = answers
		questions.append(question)
	return questions


def clean_qa_data(data_dir, category):
	input_path = "%s/QA_%s.json.gz" % (data_dir, category)
	qa_df = getDF(input_path)

	qa_df["questions"] = qa_df["questions"].apply(clean_questions)
	qa_df = qa_df[['asin', 'questions']]
	return qa_df


def clean_review(review):
	json = {}

	json["helpful"] = review["helpful"]
	json["reviewText"] = clean_text(review["reviewText"])
	return json


def clean_review_data(data_dir, category):
	input_path = "%s/reviews_%s.json.gz" % (data_dir, category)
	review_df = getDF(input_path)
	
	review_df["reviews"] = review_df[["reviewText", "helpful"]].apply(clean_review, axis=1)
	review_df = review_df.groupby("asin").apply(lambda x: x["reviews"].tolist()).reset_index()
	review_df.columns = ["asin", "reviews"]
	return review_df


def main(args):
	data_dir, category, download = args.data_dir, args.category, args.download
	
	if download == 1:
		qa_url_prefix = 'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_'
		download_data(data_dir, category, qa_url_prefix)

		review_url_prefix = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_'
		download_data(data_dir, category, review_url_prefix)


	qa_df = clean_qa_data(data_dir, category)
	reviews_df = clean_review_data(data_dir, category)
	
	qa_reviews_df = pd.merge(qa_df, reviews_df, on=['asin', 'asin'])

	output_path =  "%s/qar_products_%s.jsonl" % (data_dir, category)
	with open(output_path, 'w') as fp:
		for (_, row) in qa_reviews_df.iterrows():
			j = {}
			j['asin'] = row['asin']
			j['questions'] = row['questions']
			j['reviews'] = row['reviews']
			j['category'] = category
			fp.write(json.dumps(j) + "\n")

if __name__=="__main__":
	# parse arguments
	argParser = argparse.ArgumentParser(description="Preprocess QA and Review Data")
	argParser.add_argument("--data_dir", type=str)
	argParser.add_argument("--category", type=str)
	argParser.add_argument("--download", type=int, default=1)

	args = argParser.parse_args()
	main(args)

