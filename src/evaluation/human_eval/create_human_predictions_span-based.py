import sys
import json
import numpy as np

def read_answers_from_full(input_filename):
    all_answers = []
    print('Reading input file')
    with open(input_filename, 'r', encoding='utf-8') as fp_inp:
        for line in fp_inp:
            try:
                qar = json.loads(line)
            except json.JSONDecodeError:
                raise Exception('\"%s\" is not a valid json' % line)
            spans = [a['text'] for a in qar['answers'] if a['text'] != '']
            all_answers.append((spans, qar['human_answers']))
    return all_answers

def create_ref_and_pred(all_answers, ref_filename, prediction_filename):
    print('Writing files')
    with open(ref_filename + '_all_spans.jsonl', 'w', encoding='utf-8') as fp_ref:
        with open(prediction_filename + '_all.jsonl', 'w', encoding='utf-8') as fp_pred:
            for qid, (spans, answers) in enumerate(all_answers):
                if len(spans) > 0 and len(answers) > 0:
                    for i, span in enumerate(spans):
                        fp_pred.write('%s\n' % json.dumps({'qid': qid, 'answers': [span]}))
                        fp_ref.write('%s\n' % json.dumps({'qid': qid, 'answers': answers}))

    with open(ref_filename + '_first_span.jsonl', 'w', encoding='utf-8') as fp_ref:
        with open(prediction_filename + '_first_span.jsonl', 'w', encoding='utf-8') as fp_pred:
            for qid, (spans, answers) in enumerate(all_answers):
                if len(spans) > 0 and len(answers) > 0:
                    fp_pred.write('%s\n' % json.dumps({'qid': qid, 'answers': [spans[0]]}))
                    fp_ref.write('%s\n' % json.dumps({'qid': qid, 'answers': answers}))

    with open(ref_filename + '_all_answers.jsonl', 'w', encoding='utf-8') as fp_ref:
        with open(prediction_filename + '_all.jsonl', 'w', encoding='utf-8') as fp_pred:
            for qid, answers in enumerate(all_answers):
                if len(answers) > 1:
                    for i, answer in enumerate(answers):
                        ref_answers = answers[:i] + answers[i+1:]
                        fp_pred.write('%s\n' % json.dumps({'qid': qid, 'answers': [answer]}))
                        fp_ref.write('%s\n' % json.dumps({'qid': qid, 'answers': ref_answers}))

    with open(ref_filename + '_first_answer.jsonl', 'w', encoding='utf-8') as fp_ref:
        with open(prediction_filename + '_first.jsonl', 'w', encoding='utf-8') as fp_pred:
            for qid, answers in enumerate(all_answers):
                if len(answers) > 1:
                    fp_pred.write('%s\n' % json.dumps({'qid': qid, 'answers': [answers[0]]}))
                    fp_ref.write('%s\n' % json.dumps({'qid': qid, 'answers': answers[1:]}))

def main(input_filename, ref_filename, prediction_filename):
    all_answers = read_answers_from_full(input_filename)
    create_ref_and_pred(all_answers, ref_filename, prediction_filename)

if __name__ == '__main__':
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    main(input_filename, '%s_ref' % output_filename, '%s_pred' % output_filename)