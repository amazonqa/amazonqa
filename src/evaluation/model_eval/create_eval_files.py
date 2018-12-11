import sys
import json
import numpy as np
from tqdm import tqdm

def create_ref_and_pred(input_pred_filename, input_data_filename, ref_filename, prediction_filename):
    print('Reading file')
    try:
        with open(input_pred_filename, 'r') as fp:
            predictions = json.load(fp)
    except json.JSONDecodeError:
        raise Exception('File is not a valid json')

    predictions = dict([(int(key), value) for key, value in predictions.items()])

    print('Writing files')
    with open('%s.jsonl' % (ref_filename), 'w', encoding='utf-8') as fp_ref:
        with open('%s.jsonl' % (prediction_filename), 'w', encoding='utf-8') as fp_pred:
            with open(input_data_filename, 'r', encoding='utf-8') as fp_data:
                for line in tqdm(fp_data):
                    try:
                        qar = json.loads(line)
                    except json.JSONDecodeError:
                        raise Exception('\"%s\" is not a valid json' % line)
                    answers = [a['answerText'] for a in qar['answers'] if a['answerText'] != '']
                    qid = qar['qid']

                    if qid in predictions:
                        assert isinstance(predictions[qid], str)
                        fp_pred.write('%s\n' % json.dumps({'qid': qid, 'answers': [predictions[qid]]}))
                        fp_ref.write('%s\n' % json.dumps({'qid': qid, 'answers': answers}))

def main(input_pred_filename, input_data_filename, ref_filename, prediction_filename):
    create_ref_and_pred(input_pred_filename, input_data_filename, ref_filename, prediction_filename)

if __name__ == '__main__':
    input_pred_filename = sys.argv[1]
    input_data_filename = sys.argv[2]
    output_filename = sys.argv[3]

    main(input_pred_filename, input_data_filename, '%s_ref' % output_filename, '%s_pred' % output_filename)
