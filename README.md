# AmazonQA: A Review-Based Question Answering Task
The AmazonQA dataset is a large review-based Question Answering dataset ([paper](http://paper)). 
 
This repository comprises:
* instructions to download and work with the dataset
* implementations of preprocessing pipelines to re-generate the data for different configurations
* analyses of the dataset
* implementations of baseline models mentioned in the paper
 
# Download Instructions
The dataset can be downloaded from the following links:
* [Train](https://amazon-qa.s3-us-west-2.amazonaws.com/train-qar.jsonl)
* [Validation](https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar.jsonl)
 
# Format
The dataset is `.jsonl` format, where each line in the file is a `json` string that a question, existing answers to the question and the extracted review snippets. 
 
# Dataset Statistics
Our dataset consists of 923k questions, 3.6M answers and 14M reviews across 156k products. 
We build on the well-known Amazon dataset -  
* [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/)
* [Amazon QA Data](http://jmcauley.ucsd.edu/data/amazon/qa/)
 
Additionally, we collect additional annotations, marking each question as either answerable or unanswerable based on the available reviews.
 
## Preliminary processing scripts
The amazonqa/src/prepro/ folder contains all the scripts for downloading and preprocessing this dataset.
 
[preprocess.sh] (https://github.com/amazonqa/amazonqa/blob/master/src/prepro/preprocess_data.sh) 
