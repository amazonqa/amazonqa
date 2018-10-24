
import config
import constants as C
import json

TEMPFILEPATH = './temp'

def cat_files(params, max_review_len, seed, num_processes):
    paragraphs = []
    for process_idx in range(num_processes):
        filename='%s/squad_%s_%d_%d_%d.txt' % (TEMPFILEPATH, params[C.CATEGORY], max_review_len, seed, process_idx)
        with open(filename, 'r') as fp:
            for line in fp:
                paragraphs.append(json.loads(line.strip()))
    data = {
        'title': 'AmazonDataset',
        'paragraphs': paragraphs,
    }
    outfile = 'AmazonQA_squadformat_%s_%d_%d.json' % (params[C.CATEGORY], max_review_len, seed)
    with open(outfile, 'w') as outfile:
        json.dump(data, outfile)

def main():
    seed = 1
    max_review_len = 50
    model_name = C.LM_QUESTION_ANSWERS_REVIEWS
    params = config.get_model_params(model_name)
    main_params = config.get_main_params()
    cat_files(params, max_review_len, seed, main_params.num_processes)

if __name__ == '__main__':
    main()
