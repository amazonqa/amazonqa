import json
import gzip
import wget
import argparse
from tqdm import tqdm


def parse(path):
	g = gzip.open(path, 'r')
	for l in tqdm(g):
		yield eval(l)

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


def download_data(args):
	url_prefix = 'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_'
	#url_prefix = 'http://jmcauley.ucsd.edu/data/amazon/qa/qa_'

	url_suffix = '.json.gz'

	raw_qa_dir = args.raw_qa_dir
	cat = args.category

	download_link = url_prefix + cat + url_suffix
	wget.download(download_link, raw_qa_dir)


def clean_data(args):
	cat = args.category
	raw_qa_dir = args.raw_qa_dir
	clean_qa_dir = args.clean_qa_dir

	input_path = "%s/qa_%s.json.gz" % (raw_qa_dir, cat)
	output_path = "%s/qa_%s.jsonl" % (clean_qa_dir, cat)
	
	output_file = open(output_path, 'w')

	for l in parse(input_path):
		out_json = {}
		out_json['asin'] = l['asin']
		out_json['category'] = cat
		
		questions = []
		for q in l['questions']:
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

		out_json['questions'] = questions
		output_file.write(json.dumps(out_json)+ "\n")

	output_file.close()


if __name__=="__main__":
	# parse arguments
	argParser = argparse.ArgumentParser(description="Download and Convert Raw QA data to Clean QA data")
	argParser.add_argument("--raw_qa_dir", type=str)
	argParser.add_argument("--clean_qa_dir", type=str)
	argParser.add_argument("--category", type=str)
	argParser.add_argument("--download", type=int, default=1)

	args = argParser.parse_args()

	if args.download == 1:
		download_data(args)
	
	clean_data(args)

