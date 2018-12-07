"""
Computes the BLEU, ROUGE
using the COCO metrics scripts
"""
import json
import sys
import spacy
import argparse
import numpy as np

from collections import OrderedDict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
import pprint

from spacy.lang.en import English as NlpEnglish
nlp = spacy.load('en_core_web_lg') 
QUERY_ID_JSON_ID = 'qid'
ANSWERS_JSON_ID = 'answers'
NLP = None

def compute_evaluation_scores(reference_dict, prediction_dict, semantic=True, multiple=False, verbose=True):
    """
    reference_dict, dictionary of reference answers (qid, [answers])
    prediction_dict, dictionary of prediction answers (qid, [answer])
    score, dictionary of scores
    """

    # Check if there are missing ids in prediction
    reference_query_ids = set(reference_dict.keys())
    prediction_query_ids = set(prediction_dict.keys())
    common_query_ids = reference_query_ids.intersection(prediction_query_ids)
    assert (len(common_query_ids) == len(reference_query_ids)) and \
            (len(common_query_ids) == len(prediction_query_ids)), \
        'Reference and prediction same question ids'
    query_ids = list(reference_dict.keys())

    # Scorers
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"], 'BLEU'),
        (Rouge(), "ROUGE_L", 'ROUGE'),
        # (Meteor(),"METEOR", "METEOR"),
        (Cider(), "CIDEr", "CIDER")
    ]
    final_scores = {'min': {}, 'mean': {}, 'max': {}} if multiple else {}

    for scorer, method, method_name in scorers:
        if verbose:
            print('Computing %s..' % method_name)
        score, scores = scorer.compute_score(reference_dict, prediction_dict)
        if type(score) == list:
            if multiple:
                for m, s in zip(method, scores):
                    agg_scores = aggregate(query_ids, s)
                    for key, value in agg_scores.items():
                        final_scores[key][m] = value
            else:
                for m, s in zip(method, score):
                    final_scores[m] = s
        else:
            if multiple:
                agg_scores = aggregate(query_ids, scores)
                for key, value in agg_scores.items():
                    final_scores[key][method] = value
            else:
                final_scores[method] = score

    if semantic:
        similarities = []
        for qid, ref_answers in reference_dict.items():
            prediction_answer = nlp(prediction_dict[qid][0])
            answersimilarity = 0
            for answer in ref_answers:
                answersimilarity += prediction_answer.similarity(nlp(answer))
            similarities.append(answersimilarity / len(ref_answers))
        method = 'Semantic_Similarity'
        if multiple:
            agg_scores = aggregate(query_ids, similarities)
            for key, value in agg_scores.items():
                final_scores[key][method] = value
        else:
            final_scores[method] = np.mean(similarities)

    return final_scores

def normalize_batch(p_iter, p_batch_size=1000, p_thread_count=5):
    global NLP
    if not NLP:
        NLP = NlpEnglish(parser=False)

    output_iter = NLP.pipe(p_iter, batch_size=p_batch_size, n_threads=p_thread_count)

    for doc in output_iter:
        tokens = [str(w).strip().lower() for w in doc]
        yield ' '.join(tokens)

def load_file(filename, multiple, normalize):
    all_answers = []
    query_ids = []
    with open(filename, 'r', encoding='utf-8') as data_file:
        idx = 0
        for line in data_file:
            try:
                json_object = json.loads(line)
            except json.JSONDecodeError:
                raise Exception('\"%s\" is not a valid json' % line)

            assert QUERY_ID_JSON_ID in json_object, '\"%s\" json does not have \"%s\" field' % (line, QUERY_ID_JSON_ID)
            query_id = json_object[QUERY_ID_JSON_ID]

            assert ANSWERS_JSON_ID in json_object, '\"%s\" json does not have \"%s\" field' % (line, ANSWERS_JSON_ID)
            answers = json_object[ANSWERS_JSON_ID]
            all_answers.extend(answers)
            key = (query_id, idx) if multiple else query_id
            query_ids.extend([key]*len(answers))
            idx += 1

    all_normalized_answers = normalize_batch(all_answers) if normalize else all_answers

    query_id_to_answers_map = OrderedDict()
    for i, normalized_answer in enumerate(all_normalized_answers):
        query_id = query_ids[i]
        if query_id not in query_id_to_answers_map:
            query_id_to_answers_map[query_id] = []
        query_id_to_answers_map[query_id].append(normalized_answer)
    return query_id_to_answers_map

def compute_metrics_from_files(reference_filename, prediction_filename, multiple=False):

    reference_dictionary = load_file(reference_filename, multiple, True)
    prediction_dictionary = load_file(prediction_filename, multiple, True)

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(reference_dictionary)
    # pp.pprint(prediction_dictionary)
    # pp.pprint(list(reference_dictionary.keys()))

    for query_id, answers in prediction_dictionary.items():
        assert len(answers) <= 1, 'qid %d contains more than 1 answer \"%s\" in prediction file' % (query_id, str(answers))

    return compute_evaluation_scores(reference_dictionary, prediction_dictionary, multiple=multiple, semantic=True)

def aggregate(idxs, scores):
    d = {}
    for (qid, _), score in zip(idxs, scores):
        if qid not in d:
            d[qid] = []
        d[qid].append(score)
    return {
        'min': np.mean([np.min(value) for value in d.values()]),
        'mean': np.mean([np.mean(value) for value in d.values()]),
        'max': np.mean([np.max(value) for value in d.values()])
    }

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('path_to_reference_file')
    argparser.add_argument('path_to_prediction_file')
    argparser.add_argument("--multiple", action="store_true", default=False,)
    args = argparser.parse_args()

    metrics = compute_metrics_from_files(args.path_to_reference_file, args.path_to_prediction_file, args.multiple)
    print('############################')
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(metrics)
    if args.multiple:
        for key, value in metrics.items():
            print('## %s ##' % key)
            for metric in sorted(value):
                print('%s: %.4f' % (metric, value[metric]))
    else:
        for metric in sorted(metrics):
            print('%s: %.4f' % (metric, metrics[metric]))
    print('############################')

if __name__ == "__main__":
    main()