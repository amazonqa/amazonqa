import wget
import argparse

# Usage: python3 download_raw_qa.py --raw_qa_dir "../../data/raw_qa/ --category "Automotive"

def main(args):
	url_prefix = 'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_'
	#url_prefix = 'http://jmcauley.ucsd.edu/data/amazon/qa/qa_'

	url_suffix = '.json.gz'

	raw_qa_dir = args.raw_qa_dir
	cat = args.category

	download_link = url_prefix + cat + url_suffix
	wget.download(download_link, raw_qa_dir)


if __name__=="__main__":
	# parse arguments
	argParser = argparse.ArgumentParser(description="Download Raw QA data")
	argParser.add_argument("--raw_qa_dir", type=str)
	argParser.add_argument("--category", type=str)

	args = argParser.parse_args()

	main(args)

