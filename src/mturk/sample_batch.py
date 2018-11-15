import pandas as pd

df = pd.read_csv('data/all_100000_final.csv')
#df.drop(columns=["category" + str(i+1) for i in range(5)], inplace=True)
df.fillna('-', inplace=True)
small_df = df.iloc[0:500]
small_df.to_csv("data/all_2000_batch1_upload.csv", index=False, encoding='utf-8', mode='a')
