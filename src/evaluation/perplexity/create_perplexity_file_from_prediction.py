import sys
import json
import numpy as np
from tqdm import tqdm

def get_qar(q, answer):
    answers = [{
        'answerText': answer
    }]
    question = {}
    for key in ['questionText', 'review_snippets', 'is_answerable', 'qid']:
        question[key] = q[key]
    question['answers'] = answers
    return question

def create_file(input_pred_filename, input_data_filename, output_filename):
    print('Reading file')
    try:
        with open(input_pred_filename, 'r') as fp:
            predictions = json.load(fp)
    except json.JSONDecodeError:
        raise Exception('File is not a valid json')

    predictions = dict([(int(key), value) for key, value in predictions.items()])

    print('Writing file')
    with open('%s.jsonl' % (output_filename), 'w', encoding='utf-8') as fp_out:
        with open(input_data_filename, 'r', encoding='utf-8') as fp_data:
            for line in tqdm(fp_data):
                try:
                    qar = json.loads(line)
                except json.JSONDecodeError:
                    raise Exception('\"%s\" is not a valid json' % line)
                qid = qar['qid']

                if qid in predictions:
                    assert isinstance(predictions[qid], str)
                    fp_out.write('%s\n' % json.dumps(get_qar(qar, predictions[qid])))

if __name__ == '__main__':
    input_pred_filename = sys.argv[1]
    input_data_filename = sys.argv[2]
    output_filename = sys.argv[3]
    create_file(input_pred_filename, input_data_filename, output_filename)
