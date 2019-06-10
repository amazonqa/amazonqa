# Amazon QA
Evidence-based QA system for community question answering.

Datasets:
* [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/)
* [Amazon QA Data](http://jmcauley.ucsd.edu/data/amazon/qa/)


Language Models:

Sources - [Pytorch Word Level Language Model Example](https://github.com/pytorch/examples/tree/master/word_language_model)

# Running:
## Generating the data after pruning by length
* run utils/download\_data.py to get raw data from the website
* Open the jupyter notebook - src/notebooks/length\_percentile\_analysis
* In the first cell, change percentile to desired (Integer between [1, 100])
* Run all the cells
* In an ipython notebook- 
  * run src/utils/preprocessing.py 
  * run the method filter\_raw\_data\_all\_categories(percentile)
* the train-val-test splits would be pickled in data/input
* 95%ile is uploaded on the [drive](https://drive.google.com/open?id=17BcZcdV9vSzWchLagop8MypHTM7uFeMv)

# Related Papers

## Community/Opinion QA
* [Addressing Complex and Subjective Product-Related Queries with Customer Reviews](https://dl.acm.org/citation.cfm?id=2883044)
* [Modeling Ambiguity, Subjectivity, and Diverging Viewpoints in Opinion Question Answering Systems](https://arxiv.org/abs/1610.08095)
    * [Slide deck](https://cseweb.ucsd.edu/~m5wan/paper/icdm16_mwan_slides.pdf)
* [Answering opinion questions on products by exploiting hierarchical organization of consumer reviews](https://dl.acm.org/citation.cfm?id=2390996)
* [A survey of Community Question Answering](https://arxiv.org/abs/1705.04009)
* [Answer Selection in Community Question Answering by Normalizing Support Answers](https://link.springer.com/chapter/10.1007/978-3-319-73618-1_57)
* [Question Retrieval with High Quality Answers in Community Question Answering](https://dl.acm.org/citation.cfm?id=2661908)
* [Finding the right facts in the crowd: factoid question answering over social media](https://dl.acm.org/citation.cfm?id=1367561)

## Abstractive Summarization
* [Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond](https://arxiv.org/abs/1602.06023)
* [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
* [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182)
* [S-Net: From Answer Extraction to Answer Generation for Machine Reading Comprehension](https://arxiv.org/abs/1706.04815)
* [Dynamic Coattention Networks For Question Answering](https://arxiv.org/abs/1611.01604)
* [Machine Comprehension Using Match-LSTM and Answer Pointer](https://arxiv.org/abs/1608.07905)
* [Query Focused Abstractive Summarization](https://arxiv.org/abs/1801.07704)

## Getting Started:

### Create a virtual enviroment and install required packages
    python3 -m pip install --user virtualenv
    python3 -m virtualenv eqa_env
    source eqa_env/bin/activate
    pip3 install -r requirements.txt
    deactivate

## Example Commands:

### Setting up env
    cd src
    export PYTHONPATH=`pwd`

### Training
    nohup python3 -u main.py --category Electronics --mode train --model_name LM_QAR --hdim_a 512 --hdim_q 256 --hdim_r 256 --logfile Electronics-QAR.logfile --batch_size 10 > Electronics-QAR.out &

### Resume Training
    nohup python3 -u main.py --mode train --resume 1 --epoch 3 --save_dir saved_models/Video_Games/LM_A/2018-04-19-20-51-27 --logfile resumed_Electronics-QAR.logfile > resumed_Electronics-QAR.out &

### Testing:
    python3 -u tests/data_processing.py --model_name LM_QAR --category Dummy --batch_size 3 --max_question_len 6 --max_answer_len 5 --max_review_len 4 > out
