import argparse
import string
import math
import json

import nltk
import pandas as pd
import scipy.stats as st
from evaluator.evaluator import COCOEvalCap
from operator import itemgetter, attrgetter
from tqdm import tqdm

import retrieval_models
from nltk.corpus import stopwords


def top_reviews_and_scores(question_tokens, review_tokens, inverted_index, reviews, review_ids, select_mode, num_reviews):
    if select_mode == "random":
        scores = list(random.uniform(size=len(reviews)))
    elif select_mode in ["bm25", "indri"]:
        scores = retrieval_models.retrieval_model_scores(question_tokens, review_tokens, inverted_index, select_mode)
    elif select_mode == 'wilson':
        scores = []
        for r in range(len(reviews)):
            counts = reviews[r]['helpful']
            helpful_count = int(counts[0])
            unhelpful_count = int(counts[1]) - helpful_count

            scores.append(_wilson_score(helpful_count, unhelpful_count))
    elif select_mode == 'helpful':
        scores = [r['helpful'] for r in reviews]
    else:
        raise 'Unimplemented Review Select Mode'

    scores, top_review_ids = zip(*sorted(list(zip(scores, review_ids)), reverse=True)) if len(scores) > 0 else ([], [])
    return scores[:num_reviews], top_review_ids[:num_reviews]


def tokenize(text):
    punctuations = string.punctuation.replace("\'", '')

    for ch in punctuations:
        text = text.replace(ch, " " + ch + " ")

    tokens = text.split()
    for i, token in enumerate(tokens):
        if not token.isupper():
            tokens[i] = token.lower()
    return tokens


def get_tokens(texts, stop_words):
    text_tokens = [tokenize(r) for r in texts]
    return [[token for token in r if token not in stop_words and token not in string.punctuation] for r in text_tokens]


def find_answer_spans(args, answer_span_lens, answers, context, stop_words, question_tokens):
    context_sentences = nltk.sent_tokenize(context)
    context_text = context
    context = context.split()
    #print(context)

    sentence_tokens = get_tokens(context_sentences, stop_words)
    sentence_tokens = list(map(set, sentence_tokens))
    inverted_index = create_inverted_index(sentence_tokens)

    gold_answers_dict = {}
    gold_answers_dict[0] = [answer["answerText"] for answer in answers]
    answers_snippet_spans_bleu2 = []
    answers_snippet_spans_bleu4 = []
    answers_snippet_spans_rouge = []

    for answer_span_len in answer_span_lens:
        char_index = 0
        for word_index in range(len(context)-answer_span_len):
            span = ' '.join(context[word_index: word_index+answer_span_len])
            
            generated_answer_dict = {}
            generated_answer_dict[0] = [span]
            
            scores = COCOEvalCap.compute_scores(gold_answers_dict, generated_answer_dict)
            answers_snippet_spans_bleu2.append((scores['Bleu_2'], {
                'answer_start': char_index,
                'text': span
            }))
            answers_snippet_spans_bleu4.append((scores['Bleu_4'], {
                'answer_start': char_index,
                'text': span
            }))
            answers_snippet_spans_rouge.append((scores['ROUGE_L'], {
                'answer_start': char_index,
                'text': span
            }))

            char_index += (len(context[word_index]) + 1)

    answers_sentence_ir = []
    answers_sentence_bleu2 = []
    answers_sentence_bleu4 = []

    _, top_sentences_ir = top_reviews_and_scores(
        set(question_tokens),
        sentence_tokens,
        inverted_index,
        context_sentences,
        context_sentences,
        'bm25',
        args.span_max_num
    )

    for sentence in top_sentences_ir:
        idx = context_text.find(sentence)
        if idx >= 0:
            answers_sentence_ir.append({
                'answer_start': idx,
                'text': sentence
            })

    for sentence in context_sentences:
        bleu_scores = COCOEvalCap.compute_scores(
            gold_answers_dict,
            {0: [sentence]}
        )
        idx = context_text.find(sentence)
        if idx >= 0:
            answers_sentence_bleu2.append((bleu_scores['Bleu_2'], {
                'answer_start': idx,
                'text': sentence
            }))
            answers_sentence_bleu4.append((bleu_scores['Bleu_4'], {
                'answer_start': idx,
                'text': sentence
            }))

    answers_sentence_bleu2 = [i[1] for i in sorted(answers_sentence_bleu2, reverse=True, key=itemgetter(0))[:args.span_max_num]]
    answers_sentence_bleu4 = [i[1] for i in sorted(answers_sentence_bleu4, reverse=True, key=itemgetter(0))[:args.span_max_num]]
    answers_snippet_spans_bleu2 = [i[1] for i in sorted(answers_snippet_spans_bleu2, reverse=True, key=itemgetter(0))[:args.span_max_num]]
    answers_snippet_spans_bleu4 = [i[1] for i in sorted(answers_snippet_spans_bleu4, reverse=True, key=itemgetter(0))[:args.span_max_num]]
    answers_snippet_spans_rouge = [i[1] for i in sorted(answers_snippet_spans_rouge, reverse=True, key=itemgetter(0))[:args.span_max_num]]

    return answers_snippet_spans_bleu2, answers_snippet_spans_bleu4, answers_snippet_spans_rouge, answers_sentence_ir, answers_sentence_bleu2, answers_sentence_bleu4

def create_inverted_index(review_tokens):
    term_dict = {}
    # TODO: Use actual review IDs
    for doc_id, tokens in enumerate(review_tokens):
        for token in tokens:
            if token in term_dict:
                if doc_id in term_dict[token]:
                    term_dict[token][doc_id] += 1
                else:
                    term_dict[token][doc_id] = 1
            else:
                term_dict[token] = {doc_id: 1}
    return term_dict


def main(args):
    stop_words = set(stopwords.words('english'))
    answer_span_lens = [10, 20]
    
    wfp = open(args.output_file, 'w')
    rfp = open(args.input_file, 'r')

    for line in tqdm(rfp):
        row = json.loads(line)

        #if row["questionType"] == "yesno":
        #   continue

        if row["is_answerable"] == 1:
            reviews = row["review_snippets"]
            context = ' '.join(' '.join(reviews).split())

            answers = row["answers"]

            question_text = row["questionText"]
            question_tokens = tokenize(question_text)

            answers_snippet_spans_bleu2, answers_snippet_spans_bleu4, answers_snippet_spans_rouge, answers_sentence_ir, answers_sentence_bleu2, answers_sentence_bleu4 = find_answer_spans(args, answer_span_lens, answers, context, stop_words, question_tokens)

            qas = [{
                'id': row["qid"],
                'is_impossible': False,
                'question': row["questionText"],
                'answers_snippet_spans_bleu2': answers_snippet_spans_bleu2,
                'answers_snippet_spans_bleu4': answers_snippet_spans_bleu4,
                'answers_snippet_spans_rouge': answers_snippet_spans_rouge,
                'answers_sentence_ir': answers_sentence_ir,
                'answers_sentence_bleu2': answers_sentence_bleu2,
                'answers_sentence_bleu4': answers_sentence_bleu4,
                'human_answers': [answer["answerText"] for answer in answers],
            }]

            wfp.write(json.dumps({
                'context': context,
                'qas': qas,
            }) + '\n')
    wfp.close()


if __name__ == '__main__':
    # parse arguments
    argParser = argparse.ArgumentParser(description="Convert Amazon QAR to Squad format")
    argParser.add_argument("--input_file", type=str)
    argParser.add_argument("--output_file", type=str)
    argParser.add_argument("--span_max_num", type=int, default=5)
    argParser.add_argument("--evaluation_metric", type=str, default="Bleu_2")

    args = argParser.parse_args()
    main(args)

