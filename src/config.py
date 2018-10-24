import argparse
import constants as C

def debugprint(*args):
    print(args)

def get_main_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest='model_name', type=str, default=C.LM_ANSWERS)
    parser.add_argument('--mode', dest='mode', type=str, default=C.TRAIN_TYPE)
    parser.add_argument('--input_path', dest='input_path', type=str, default=None)
    parser.add_argument('--epoch', dest='epoch', type=int, default=C.BEST_EPOCH_IDX)
    parser.add_argument('--resume', dest='resume', type=int, default=0)
    parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output.txt')
    parser.add_argument('--process_idx', dest='process_idx', type=int, default=0)
    parser.add_argument('--num_processes', dest='num_processes', type=int, default=1)
    args, _ = parser.parse_known_args()
    return args

def get_model_params(model_name):
    H = _model_hyperparams(C.LM_MODELS, C.LM_HP)[model_name]
    parser = argparse.ArgumentParser()

    add_arg(parser, str, C.MODEL_NAME, H)
    add_arg(parser, str, C.CATEGORY, H)
    add_arg(parser, str, C.REVIEW_SELECT_MODE, H)
    add_arg(parser, int, C.REVIEW_SELECT_NUM, H)
    add_arg(parser, int, C.NUM_EPOCHS, H)
    add_arg(parser, int, C.BATCH_SIZE, H)
    add_arg(parser, float, C.DROPOUT, H)
    add_arg(parser, float, C.LR, H)
    add_arg(parser, int, C.HDIM_A, H)
    add_arg(parser, int, C.HDIM_R, H)
    add_arg(parser, int, C.HDIM_Q, H)
    add_arg(parser, int, C.EMBEDDING_DIM, H)
    add_arg(parser, int, C.H_LAYERS, H)
    add_arg(parser, int, C.DECAY_START_EPOCH, H)
    add_arg(parser, float, C.LR_DECAY, H)
    add_arg(parser, float, C.GLOBAL_NORM_MAX, H)
    add_arg(parser, int, C.VOCAB_SIZE, H)
    add_arg(parser, float, C.TEACHER_FORCING_RATIO, H)
    add_arg(parser, int, C.OUTPUT_MAX_LEN, H)
    add_arg(parser, str, C.LOG_FILENAME, H)
    add_arg(parser, str, C.OPTIMIZER_TYPE, H)
    add_arg(parser, int, C.MAX_QUESTION_LEN, H)
    add_arg(parser, int, C.MAX_ANSWER_LEN, H)
    add_arg(parser, int, C.MAX_REVIEW_LEN, H)
    add_arg(parser, str, C.SAVE_DIR, H)
    add_arg(parser, int, C.USE_ATTENTION, H)

    args, _ = parser.parse_known_args()
    return vars(args)

def add_arg(parser, typ, hpstr, H):
    parser.add_argument('--' + hpstr, dest=hpstr, type=typ, default=H[hpstr])

def _model_hyperparams(keys, values):
    d = {}
    for itr, key in enumerate(keys):
        d[key] = {}
        for h_key, val in values.items():
            d[key][h_key] = val[itr]
    return d
