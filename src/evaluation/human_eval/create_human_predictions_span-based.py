import sys
import json
import numpy as np

SPAN_KEYS = [
    'answers_sentence_ir',
    'answers_sentence_bleu2',
    'answers_sentence_bleu4',
    'answers_snippet_spans_bleu2',
    'answers_snippet_spans_bleu4',
    'answers_snippet_spans_rouge',
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
            qar = qar['qas'][0]
            spans = {}
            for key in SPAN_KEYS:
                spans[key] = [a['text'] for a in qar[key] if a['text'] != '']
            all_answers.append((spans, qar['human_answers']))
    return all_answers

def create_ref_and_pred(all_answers, ref_filename, prediction_filename):
    print('Writing files')
    for span_key in SPAN_KEYS:
        with open('%s_spans_%s.jsonl' % (ref_filename, span_key), 'w', encoding='utf-8') as fp_ref:
            with open('%s_spans_%s.jsonl' % (prediction_filename, span_key), 'w', encoding='utf-8') as fp_pred:
                for qid, (spans, answers) in enumerate(all_answers):
                    if len(spans[span_key]) > 0 and len(answers) > 0:
                        for i, span in enumerate(spans[span_key]):
                            fp_pred.write('%s\n' % json.dumps({'qid': qid, 'answers': [span]}))
                            fp_ref.write('%s\n' % json.dumps({'qid': qid, 'answers': answers}))

    with open(ref_filename + '_human_answers_all.jsonl', 'w', encoding='utf-8') as fp_ref:
        with open(prediction_filename + '_human_answers_all.jsonl', 'w', encoding='utf-8') as fp_pred:
            for qid, (spans, answers) in enumerate(all_answers):
                if len(answers) > 1:
                    for i, answer in enumerate(answers):
                        ref_answers = answers[:i] + answers[i+1:]
                        fp_pred.write('%s\n' % json.dumps({'qid': qid, 'answers': [answer]}))
                        fp_ref.write('%s\n' % json.dumps({'qid': qid, 'answers': ref_answers}))

def main():
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    ref_filename, prediction_filename = '%s_ref' % output_filename, '%s_pred' % output_filename
    all_answers = read_answers_from_full(input_filename)
    create_ref_and_pred(all_answers, ref_filename, prediction_filename)

if __name__ == '__main__':
    main()
