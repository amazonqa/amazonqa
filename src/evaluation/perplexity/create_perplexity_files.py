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

def create_files(qar_filename, squad_filename, output_filename):
    fp_helpful = open('%s_review_helpful.jsonl' % output_filename, 'w')
    fp_wilson = open('%s_review_wilson.jsonl' % output_filename, 'w')
    fp_sent_rand = open('%s_sent_rand.jsonl' % output_filename, 'w')
    fp_sent_ir = open('%s_sent_ir.jsonl' % output_filename, 'w')
    # fp_sent_bleu = open('%s_sent_bleu' % output_filename, 'w')

    print('Writing files')
    with open(qar_filename, 'r', encoding='utf-8') as fp_data:
        for line in tqdm(fp_data):
            try:
                qar = json.loads(line)
            except json.JSONDecodeError:
                raise Exception('\"%s\" is not a valid json' % line)
            if qar['is_answerable'] == 0:
                continue
            
            fp_helpful.write('%s\n' % json.dumps(get_qar(qar, qar['top_review_helpful'][0])))
            fp_wilson.write('%s\n' % json.dumps(get_qar(qar, qar['top_review_wilson'][0])))
            fp_sent_rand.write('%s\n' % json.dumps(get_qar(qar, qar['random_sentence'][0])))
            fp_sent_ir.write('%s\n' % json.dumps(get_qar(qar, qar['top_sentences_IR'][0])))

    fp_helpful.close()
    fp_wilson.close()
    fp_sent_rand.close()
    fp_sent_ir.close()

if __name__ == '__main__':
    qar_filename = sys.argv[1]
    squad_filename = sys.argv[2]
    output_filename = sys.argv[3]
    create_files(qar_filename, squad_filename, output_filename)
