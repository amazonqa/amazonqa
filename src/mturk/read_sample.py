import pandas as pd
import numpy as np
import random
import math
import sys
from tqdm import tqdm

num_ques = 5
num_snips = 5

seed = 89
random.seed(seed)
np.random.seed(seed)

# Dropping old Name columns 
# reviews_df = df["reviews_q"].str.strip().str.split("1\\)|2\\)|3\\)|4\\)|5\\)", expand = True)
#for i in range(1, num_snips + 1):
#	df["Snippet" + str(i)] = reviews_df[i]

#df.drop(columns =["reviews_q"], inplace = True)
#reviews_df = None

#id	question	reviews_q	is_answerable	is_easy
easy_df = pd.read_csv(sys.argv[2] + '.csv')
easy_df.drop(columns = ['is_answerable', 'relevant_snippets'], inplace=True)

reg_df = pd.read_csv(sys.argv[1] + '.csv')
#SORT BY CATEGORY
print(reg_df.columns)
reg_df.sort_values(by=["category"], inplace=True)

for i in range(num_snips):
	reg_df["Snippet" + str(i+1)] = reg_df["review" + str(i)]

reg_df.drop(columns = ["category"] + ["review" + str(i) for i in range(num_ques)], inplace=True)


print(easy_df.columns)
print(reg_df.columns)

#assert(set(easy_df.columns) == set(reg_df.columns))

df_columns = []
for i in range(1, num_ques + 1):
	df_columns.append('id' + str(i))
	df_columns.append('question' + str(i))
	for j in range(1, num_snips + 1):
		df_columns.append('Snippet' + str(i) + str(j))

num_hits = math.floor(len(reg_df)/ (num_ques - 1)*1.)

def convert_to_row(hits, easy_pos):
	#print(len(hits), '\n')
	d = {}
	for i in range(num_ques):
		hit = hits.iloc[i]
		d['id' + str(i+1)] = hit['id']
		d['question' + str(i+1)] = hit['question']
		for j in range(num_snips):
			d['Snippet' + str(i+1) + str(j+1)] = hit['Snippet' + str(j+1)]
			if j == easy_pos:
				d['is_easy' + str(j+1)] = 'Y'
			else:
				d['is_easy' + str(j+1)] = 'N'
	row = pd.Series(d)
	return row

easy_pos = 0
reg_pos = 0
easy_ind = 0
def get_next_hit():
	global easy_pos, reg_pos, easy_ind
	num_reg = num_ques - 1
	#print(reg_pos, easy_pos, easy_pos, reg_pos + num_reg)
	def1 = reg_df.iloc[reg_pos : reg_pos + easy_pos]
	def2 = pd.DataFrame(easy_df[easy_ind : easy_ind+1])
	def3 = reg_df.iloc[reg_pos + easy_pos: reg_pos + num_reg]
	yield convert_to_row( pd.concat([def1, def2, def3], ignore_index=True), easy_pos)
	reg_pos += num_reg
	easy_pos = (easy_pos + 1) % num_ques
	easy_ind = (easy_ind + 1) % len(easy_df)


all_rows = []
for hit in tqdm(range(num_hits)):
	all_rows.append(pd.DataFrame(get_next_hit(), columns=df_columns))

hits_df = pd.concat(all_rows, ignore_index=True)

#SHUFFLE HITS
hits_df = hits_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

hits_df.to_csv(sys.argv[1] + '_final.csv', index=False)

