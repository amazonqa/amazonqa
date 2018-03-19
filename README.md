# Explicable-Question-Answering
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
* 95%ile is uploaded on the [drive] (https://drive.google.com/open?id=1e7JvTLbQObfV_MrdbgGdnxeuH0cMN_xP)

