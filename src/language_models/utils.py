import argparse
import constants as C

def debugprint(*args):
    print(args)

def get_model_params(model_type):
    H = C.LM_HP[model_type]
    parser = argparse.ArgumentParser()
    add_arg(parser, int, C.EPOCHS, H)
    add_arg(parser, int, C.BATCH_SIZE, H)
    add_arg(parser, int, C.HDIM, H)
    add_arg(parser, float, C.DROPOUT, H)
    add_arg(parser, float, C.LR, H)
    add_arg(parser, int, C.DECAY_START_EPOCH, H)
    add_arg(parser, float, C.GLOBAL_NORM_MAX, H)
    add_arg(parser, int, C.H_LAYERS, H)
    add_arg(parser, int, C.VOCAB_SIZE, H)
    add_arg(parser, int, C.TRAIN_LINES, H)
    params = vars(parser.parse_args())
    params[C.MODEL_NAME] = model_type
    return params

def add_arg(parser, typ, hpstr, H):
    parser.add_argument('--' + hpstr, dest=hpstr, type=typ, default=H[hpstr])
