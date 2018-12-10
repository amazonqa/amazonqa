import sys
import json
import numpy as np

KEYS = [
    'random_sentence',
    'top_sentences_IR',
    'top_review_wilson',
    'top_review_helpful',
]

def read_answers_from_full(input_filename):
    all_answers = []
    print('Reading input file')
    with open(input_filename, 'r', encoding='utf-8') as fp_inp:
        for line in fp_inp:
            try:
                qar = json.loads(line)
            except json.JSONDecodeError:
                raise Exception('\"%s\" is not a valid json' % line)
            reviews = {}
            for key in KEYS:
                reviews[key] = qar[key]
            reviews['top_sentences_IR'] = reviews['top_sentences_IR'][:5]
            answers = [a['answerText'] for a in qar['answers'] if a['answerText'] != '']
            all_answers.append((reviews, answers))
    return all_answers

def create_ref_and_pred(all_answers, ref_filename, prediction_filename):
    print('Writing files')

    for key in KEYS:
        with open('%s_%s.jsonl' % (ref_filename, key), 'w', encoding='utf-8') as fp_ref:
            with open('%s_%s.jsonl' % (prediction_filename, key), 'w', encoding='utf-8') as fp_pred:
                for qid, (reviews, answers) in enumerate(all_answers):
                    if len(answers) > 1:
                        for i, review in enumerate(reviews[key]):
                            fp_pred.write('%s\n' % json.dumps({'qid': qid, 'answers': [review]}))
                            fp_ref.write('%s\n' % json.dumps({'qid': qid, 'answers': answers}))

def main(input_filename, ref_filename, prediction_filename):
    all_answers = read_answers_from_full(input_filename)
    create_ref_and_pred(all_answers, ref_filename, prediction_filename)

if __name__ == '__main__':
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    main(input_filename, '%s_ref' % output_filename, '%s_pred' % output_filename)
