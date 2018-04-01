import argparse
import constants as C

def debugprint(*args):
    print(args)

def get_model_params(model_name):
    H = _model_hyperparams(C.LM_MODELS, C.LM_HP)[model_name]
    parser = argparse.ArgumentParser()
    add_arg(parser, int, C.EPOCHS, H)
    add_arg(parser, int, C.BATCH_SIZE, H)
    add_arg(parser, float, C.DROPOUT, H)
    add_arg(parser, float, C.LR, H)
    add_arg(parser, int, C.HDIM, H)
    add_arg(parser, int, C.H_LAYERS, H)
    add_arg(parser, int, C.DECAY_START_EPOCH, H)
    add_arg(parser, int, C.LR_DECAY, H)
    add_arg(parser, float, C.GLOBAL_NORM_MAX, H)
    add_arg(parser, int, C.VOCAB_SIZE, H)
    add_arg(parser, float, C.TEACHER_FORCING_RATIO, H)
    add_arg(parser, int, C.OUTPUT_MAX_LEN, H)
    add_arg(parser, str, C.CATEGORY, H)
    add_arg(parser, str, C.MODEL_NAME, H)
    add_arg(parser, str, C.LOG_FILENAME, H)
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
