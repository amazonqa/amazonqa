import wget
from constants import *


url_prefix = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_'
url_suffix = '_5.json.gz'


for cat in CATEGORIES:
    wget.download(url_prefix + cat + url_suffix)

url_prefix = 'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_'
url_suffix = '.json.gz'

for cat in CATEGORIES:
    wget.download(url_prefix + cat + url_suffix)
