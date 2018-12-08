import pandas as pd
import sys
import json

def load_data(input_filename, n_questions=400):
    entries = []
    print('Reading input file')
    i = 0
    with open(input_filename, 'r', encoding='utf-8') as fp_inp:
        for line in fp_inp:
            try:
                qar = json.loads(line)
            except json.JSONDecodeError:
                raise Exception('\"%s\" is not a valid json' % line)
            if i == n_questions:
                break
            context = qar['context']
            qar = qar['qas'][0]
            spans = [a['text'] for a in qar['answers'] if a['text'] != '']
            question = qar['question']
            answers = qar['human_answers']
            entries.append({
                'question': question,
                'context': context,
                'answers': answers,
                'spans': spans
            })
            i += 1
    return entries

def main():
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    print('Writing output file')
    data = pd.DataFrame(load_data(input_filename))
    data[['question', 'context', 'spans', 'answers']].to_csv(output_filename)

if __name__ == '__main__':
    main()
