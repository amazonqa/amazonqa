"""
Computes the BLEU, ROUGE
using the COCO metrics scripts
"""
import json
import sys
import spacy

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

from spacy.lang.en import English as NlpEnglish
nlp = spacy.load('en_core_web_lg') 
QUERY_ID_JSON_ID = 'qid'
ANSWERS_JSON_ID = 'answers'
NLP = None

def compute_evaluation_scores(reference_dict, prediction_dict, semantic=True, verbose=True):
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

    # Scorers
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"], 'BLEU'),
        (Rouge(), "ROUGE_L", 'ROUGE'),
        # (Meteor(),"METEOR", "METEOR"),
        # (Cider(), "CIDEr", "CIDER")
    ]
    final_scores = {}
    for scorer, method, method_name in scorers:
        if verbose:
            print('Computing %s..' % method_name)
        score, _ = scorer.compute_score(reference_dict, prediction_dict)
        # score, scorers = scorer.compute_score(reference_dict, prediction_dict)
        # print(scorers)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    if semantic:
        similarity = 0
        for qid, ref_answers in reference_dict.items():
            prediction_answer = nlp(prediction_dict[qid][0])
            answersimilarity = 0
            for answer in ref_answers:
                answersimilarity += prediction_answer.similarity(nlp(answer))
            similarity += (answersimilarity / len(ref_answers))
        semantic_similarity = similarity / len(reference_dict)
        final_scores['Semantic_Similarity'] = semantic_similarity

    return final_scores

def normalize_batch(p_iter, p_batch_size=1000, p_thread_count=5):
    global NLP
    if not NLP:
        NLP = NlpEnglish(parser=False)

    output_iter = NLP.pipe(p_iter, batch_size=p_batch_size, n_threads=p_thread_count)

    for doc in output_iter:
        tokens = [str(w).strip().lower() for w in doc]
        yield ' '.join(tokens)

def load_file(filename, normalize):
    all_answers = []
    query_ids = []
    with open(filename, 'r', encoding='utf-8') as data_file:
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
            query_ids.extend([query_id]*len(answers))

    all_normalized_answers = normalize_batch(all_answers) if normalize else all_answers

    query_id_to_answers_map = {}
    for i, normalized_answer in enumerate(all_normalized_answers):
        query_id = query_ids[i]
        query_id_to_answers_map[query_id] = []
        query_id_to_answers_map[query_id].append(normalized_answer)
    return query_id_to_answers_map

def compute_metrics_from_files(reference_filename, prediction_filename, normalize=False):

    reference_dictionary = load_file(reference_filename, normalize)
    prediction_dictionary = load_file(prediction_filename, normalize)

    for query_id, answers in prediction_dictionary.items():
        assert len(answers) <= 1, 'qid %d contains more than 1 answer \"%s\" in prediction file' % (query_id, str(answers))

    return compute_evaluation_scores(reference_dictionary, prediction_dictionary, semantic=True)

def main():
    path_to_reference_file = sys.argv[1]
    path_to_prediction_file = sys.argv[2]

    metrics = compute_metrics_from_files(path_to_reference_file, path_to_prediction_file)
    print('############################')
    for metric in sorted(metrics):
        print('%s: %.4f' % (metric, metrics[metric]))
    print('############################')

if __name__ == "__main__":
    main()
