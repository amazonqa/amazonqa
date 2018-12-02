import random
from tqdm import tqdm
import spacy
import json
import numpy as np
import os.path
import pickle
import argparse

nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def main(args):
    wfp_eval = open(args.eval_file, 'w')
    wfp = open(args.examples_file, 'w')
    
    rfp = open(args.file, 'r')

    for line in tqdm(rfp):
        para = json.loads(line)
        context = para["context"]
        context_tokens = context.split()
        context_chars = [list(token) for token in context_tokens]
        spans = convert_idx(context, context_tokens)
        for qa in para["qas"]:
            ques = qa["question"].replace(
                "''", '" ').replace("``", '" ')
            ques_tokens = ques.split()
            ques_chars = [list(token) for token in ques_tokens]
            
            y1s, y2s = [], []
            answer_texts = []
            for answer in qa["answers"]:
                answer_text = answer["text"]
                answer_start = answer['answer_start']
                answer_end = answer_start + len(answer_text)
                answer_texts.append(answer_text)
                answer_span = []
                for idx, span in enumerate(spans):
                    if not (answer_end <= span[0] or answer_start >= span[1]):
                        answer_span.append(idx)
                y1, y2 = answer_span[0], answer_span[-1]
                y1s.append(y1)
                y2s.append(y2)
            
            example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                   "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": qa["_id"]}
            wfp.write(json.dumps(example) + '\n')

            row = {"context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"], "id": qa["_id"]}
            wfp_eval.write(json.dumps(row) + '\n')

    rfp.close()
    wfp_eval.close()
    wfp.close()


if __name__ == '__main__':
    # parse arguments
    argParser = argparse.ArgumentParser(description="Convert Squad to Post-Squad format")
    argParser.add_argument("--file", type=str)
    argParser.add_argument("--eval_file", type=str)
    argParser.add_argument("--examples_file", type=str)

    args = argParser.parse_args()
    main(args)

