
import convert_squad
import config
import constants as C
import json

TEMPFILEPATH = './temp'

def cat_files(category, mode, max_review_len, max_num_spans, max_num_products, seed, num_processes):
    paragraphs = []
    for process_idx in range(num_processes):
        filename = convert_squad.process_filepath(category, mode, max_review_len, max_num_spans, seed, process_idx)
        with open(filename, 'r') as fp:
            for line in fp:
                paragraphs.append(json.loads(line.strip()))
    data = [{
        'title': 'AmazonDataset',
        'paragraphs': paragraphs,
    }]
    
    out = {"data":data, "version":"1.0"}

    outfile = 'Amazon-Squad_%s_%s_%d_%d_%d_%d.json' % (category, mode, max_review_len, max_num_spans, max_num_products, seed)
    with open(outfile, 'w') as outfile:
        json.dump(out, outfile)

def main():
    main_params = convert_squad.get_main_params()
    model_name = C.LM_QUESTION_ANSWERS_REVIEWS
    params = config.get_model_params(model_name)
    params[C.MODEL_NAME] = model_name 

    model_name = C.LM_QUESTION_ANSWERS_REVIEWS
    params = config.get_model_params(model_name)
    cat_files(
        params[C.CATEGORY], 
        main_params.mode, 
        main_params.max_review_len, 
        main_params.max_num_spans, 
        main_params.max_num_products, 
        main_params.seed, 
        main_params.num_processes
    )

if __name__ == '__main__':
    main()

