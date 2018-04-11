import argparse
import constants as C
import scipy.stats as st
import math
import random

def debugprint(*args):
    print(args)

def get_main_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest='model_name', type=str, default=C.LM_ANSWERS)
    parser.add_argument('--mode', dest='mode', type=str, default=C.TRAIN_TYPE)
    args, _ = parser.parse_known_args()
    return args.model_name, args.mode

def get_model_params(model_name):
    H = _model_hyperparams(C.LM_MODELS, C.LM_HP)[model_name]
    parser = argparse.ArgumentParser()
    add_arg(parser, str, C.MODEL_NAME, H)
    add_arg(parser, str, C.CATEGORY, H)
    add_arg(parser, str, C.REVIEW_SELECT_MODE, H)
    add_arg(parser, int, C.REVIEW_SELECT_NUM, H)
    add_arg(parser, int, C.EPOCHS, H)
    add_arg(parser, int, C.BATCH_SIZE, H)
    add_arg(parser, float, C.DROPOUT, H)
    add_arg(parser, float, C.LR, H)
    add_arg(parser, int, C.HDIM, H)
    add_arg(parser, int, C.EMBEDDING_DIM, H)
    add_arg(parser, int, C.H_LAYERS, H)
    #add_arg(parser, int, C.DECAY_START_EPOCH, H)
    add_arg(parser, int, C.LR_DECAY, H)
    add_arg(parser, float, C.GLOBAL_NORM_MAX, H)
    add_arg(parser, int, C.VOCAB_SIZE, H)
    add_arg(parser, float, C.TEACHER_FORCING_RATIO, H)
    add_arg(parser, int, C.OUTPUT_MAX_LEN, H)
    add_arg(parser, str, C.LOG_FILENAME, H)
    add_arg(parser, int, C.MAX_QUESTION_LEN, H)
    add_arg(parser, int, C.MAX_ANSWER_LEN, H)
    add_arg(parser, int, C.MAX_REVIEW_LEN, H)
    return vars(parser.parse_args())

def add_arg(parser, typ, hpstr, H):
    parser.add_argument('--' + hpstr, dest=hpstr, type=typ, default=H[hpstr])

def _model_hyperparams(keys, values):
    d = {}
    for itr, key in enumerate(keys):
        d[key] = {}
        for h_key, val in values.items():
            d[key][h_key] = val[itr]
    return d

def select_reviews(reviews, select_mode, num_reviews):
    if select_mode == C.RANDOM:
        random.shuffle(reviews)
    elif select_mode == C.WILSON:
        for r in range(len(reviews)):
            helpful_count = reviews[r]['helpful']
            # TODO: the 'unhelpful' key is wrong in the database! should be 'total'
            unhelpful_count = reviews[r]['unhelpful'] - helpful_count
            reviews[r]['wilson_score'] = wilson_score(helpful_count, unhelpful_count)
        reviews = sorted(reviews, key=lambda review: review['wilson_score'], reverse=True)
    else: #select_mode == HELPFUL or default 
        reviews = sorted(reviews, key=lambda review: review['helpful'], reverse=True)
    return reviews[:num_reviews]

def wilson_score(positive, negative):
    confidence = 0.98
    n = positive + negative
    if n == 0: return 0.
    phat = (1.*positive) / n
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    return (phat + z*z/(2*n) - z * math.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)
