import argparse
import string
import math
import json

import pandas as pd
from tqdm import tqdm

def main(args):
	wfp = open(args.output_file, 'w')
	rfp = open(args.input_file, 'r')
	qid = 0

	for line in tqdm(rfp):
		row = json.loads(line)
		row['qas'][0]['qid'] = qid
		
		wfp.write(json.dumps(row) + '\n')
		qid += 1

	wfp.close()


if __name__ == '__main__':
	# parse arguments
	argParser = argparse.ArgumentParser(description="Convert Amazon QAR to Squad format")
	argParser.add_argument("--input_file", type=str)
	argParser.add_argument("--output_file", type=str)

	args = argParser.parse_args()
	main(args)

