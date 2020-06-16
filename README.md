# AmazonQA: A Review-Based Question Answering Task
The AmazonQA dataset is a large review-based Question Answering dataset ([paper](https://arxiv.org/abs/1908.04364)). 
 
This repository comprises:
* instructions to download and work with the dataset
* implementations of preprocessing pipelines to re-generate the data for different configurations
* analyses of the dataset
* implementations of baseline models mentioned in the paper

# Updates
06/16 - We have uploaded the test sets for all the dataset formats below.

# Download Instructions
The dataset can be downloaded from the following links:
* [Train](https://amazon-qa.s3-us-west-2.amazonaws.com/train-qar.jsonl)
* [Validation](https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar.jsonl)
* [Test](https://drive.google.com/file/d/1A_gaYbyBUOfwi8CQ7d5OO_b91lEvSnwr/view?usp=sharing)
 
# Format
The dataset is `.jsonl` format, where each line in the file is a `json` string that corresponds to a question, existing answers to the question and the extracted review snippets (relevant to the question).

Each `json` string has many fields. Here are the fields that the QA training pipeline uses:

* questionText: String. The question.
* questionType: String. Either "yesno" for a boolean question, or "descriptive" for a non-boolean question.
* review_snippets: List of strings. Extracted review snippets relevant to the question (at most ten). 
* answers: List of dicts, one for each answer. Each dict has the following fields. 
  * answerText: String. The text for the answer.
  * answerType: String. Type of the answer.
  * helpful: List of two integers. The first integer indicates the number of uses who found the answer helpful. The second integer indicates the total number of responses.

Here are some other fields that we use for evaluation and analysis:
* asin: String. Unique product ID for the product the question pertains to.
* qid: Integer. Unique question id for the question (in the entire dataset).
* category: String. Product category.
* top_review_wilson: String. The review with the highest wilson score.
* top_review_helpful: String. The review voted as most helpful by the users.
* is_answerable: boolean. Output of the answerability classifier indicating whether the question is answerable using the review snippets. 
* top_sentences_IR: List of strings. A list of top sentences (at most 10) based on IR score with the question. 
 
# Dataset Statistics
Our dataset consists of 923k questions, 3.6M ansheers and 14M reviews across 156k products. 
We build on the well-known Amazon dataset -  
* [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/)
* [Amazon QA Data](http://jmcauley.ucsd.edu/data/amazon/qa/)
 
Additionally, we collect additional annotations, marking each question as either answerable or unanswerable based on the available reviews.
 
# Data Processing
 
## Scripts
The src/prepro/ folder contains all the scripts for generating raw and different processed datsets.
 
## Raw Products Dataset
The [script](https://github.com/amazonqa/amazonqa/blob/master/src/prepro/preprocess_data.sh) generates the raw train/val/test product splits by combining the well known amazon reviews and questions dataset for all the categories.
 
[train](https://amazon-qa.s3-us-west-2.amazonaws.com/train-qar_products.jsonl)
[val](https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar_products.jsonl)
[test](https://drive.google.com/file/d/1g2YYhtX5Te665-dlKssGHn5AZCrvyfYc/view?usp=sharing)

## Processed Dataset
The [script](https://github.com/amazonqa/amazonqa/blob/master/src/prepro/create_data.sh) creates question-answers pairs with query-relevant review snippets and is_answerable annotation by a trained classifier. More details regarding this step are mentioned in the section 3.1 Data Processing.
 
[train](https://amazon-qa.s3-us-west-2.amazonaws.com/train-qar.jsonl)
[val](https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar.jsonl)
[test](https://drive.google.com/file/d/1A_gaYbyBUOfwi8CQ7d5OO_b91lEvSnwr/view?usp=sharing)

## Auxilliary Datasets
We also provide the scripts to convert our dataset to other question answering dataset formats like squad and ms-marco.
 
### Span-based 
The [script](https://github.com/amazonqa/amazonqa/blob/master/src/prepro/convert_squad.sh) converts our dataset to squad format by extracting snippets using different span-heuristics. More details regarding this step are mentioned in the section 5.2 Span-based QA model.

[train](https://amazon-qa.s3-us-west-2.amazonaws.com/train-qar_squad.jsonl)
[val](https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar_squad.jsonl)
[test](https://drive.google.com/file/d/1eede6X_r7uoOmDZkv5NlbM4Mu-OP-cCe/view?usp=sharing)

### Generative
The [script](https://github.com/amazonqa/amazonqa/blob/master/src/prepro/convert_msmarco.sh) converts our dataset MSMARCO format.

[train](https://amazon-qa.s3-us-west-2.amazonaws.com/train-qar_msmarco.jsonl)
[val](https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar_msmarco.jsonl)
[test](https://drive.google.com/file/d/13wMbyP__PEaH61Dsy5R--kreJtU1u9Ue/view?usp=sharing)

### Answerability Classifier
Binary classifier and related files can be found at [link](https://amazon-qa.s3-us-west-2.amazonaws.com/answerability_classifier.zip)
