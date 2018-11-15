import pandas as pd
import numpy as np
import random
import math

num_ques = 5
num_snips = 5

seed = 0
random.seed(seed)
np.random.seed(seed)


# Dropping old Name columns 
df = pd.read_csv('gold_set_raw.csv')
reviews_df = df["reviews_q"].str.strip().str.split("1\\)|2\\)|3\\)|4\\)|5\\)", expand = True)
for i in range(1, num_snips + 1):
	df["Snippet" + str(i)] = reviews_df[i]

df.drop(columns =["reviews_q"], inplace = True)
reviews_df = None


#id	question	reviews_q	is_answerable	is_easy
easy_df = df.loc[df['is_easy'] == 'easy']
easy_df_save = easy_df.drop(columns=['is_easy'])
easy_df_save.to_csv("data/easy_data.csv", index=False)
exit(0)
reg_df = df.loc[df['is_easy'] != 'easy']

df_columns = []
extra_columns = []
for i in range(1, num_ques + 1):
	df_columns.append('id' + str(i))
	df_columns.append('question' + str(i))
	extra_columns.append('is_answerable' + str(i))
	extra_columns.append('is_easy' + str(i))
	extra_columns.append('relevant_snippets' + str(i))
	for j in range(1, num_snips + 1):
		df_columns.append('Snippet' + str(i) + str(j))

num_hits = math.ceil(len(df) / 5.0)
print(len(df), num_hits)

def convert_to_row(hits, easy_pos):
	#print(len(hits), '\n')
	d = {}
	for i in range(num_ques):
		hit = hits.iloc[i]
		d['id' + str(i+1)] = hit['id']
		d['question' + str(i+1)] = hit['question']
		d['is_answerable' + str(i+1)] = hit['is_answerable']
		d['relevant_snippets' + str(i+1)] = hit['relevant_snippets']
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

hits_df = pd.concat([pd.DataFrame(get_next_hit(), columns=df_columns + extra_columns) for hit in range(num_hits)], ignore_index=True)
hits_df.to_csv('gold_set_mturk_with_ans.csv', index=False)

hits_df.drop(columns = extra_columns, inplace = True)
hits_df.to_csv('gold_set_mturk.csv', index=False)
print(pd.read_csv('gold_set_mturk.csv').columns)
