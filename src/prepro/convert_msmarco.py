import argparse
import json
from tqdm import tqdm


def main(args):
	wfp = open(args.output_file, 'w')
	rfp = open(args.input_file, 'r')

	for line in tqdm(rfp):
		row = json.loads(line)

		if row["is_answerable"] == 1:
			reviews = row["review_snippets"]
			passages = []
			for review in reviews:
				passage = {}
				passage["is_selected"] = 1
				passage["url"] = ""
				passage["passage_text"] = review
				passages.append(passage)

			answers = row["answers"]

			final_json = {}
			final_json["answers"] = [answer["answerText"] for answer in answers]
			final_json["passages"] = passages
			final_json["query"] = row["questionText"]
			final_json["query_id"] = row["qid"]
			
			if row["questionType"] == "descriptive":
				final_json["query_type"] = "DESCRIPTION"
			elif row["questionType"] == "yesno":
				final_json["query_type"] = "YESNO"
			else:
				print("new query_type "+ row)
				exit(0)

			final_json["wellFormedAnswers"] = []

			wfp.write(json.dumps(final_json) + '\n')

	wfp.close()


if __name__ == '__main__':
	# parse arguments
	argParser = argparse.ArgumentParser(description="Convert Amazon QAR to MSMARCO format")
	argParser.add_argument("--input_file", type=str)
	argParser.add_argument("--output_file", type=str)

	args = argParser.parse_args()
	main(args)

