"""
Computes the BLEU, ROUGE
using the COCO metrics scripts
"""
from bleu.bleu import Bleu
from rouge.rouge import Rouge
from tokenizer.ptbtokenizer import PTBTokenizer

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

if __name__ == '__main__':
    hypo = {0: ['Hello BLEU hypothesis']}
    ref = {0: ['Hello BLEU hypothesis', 'Hello BLEU hypothesis']}
    
    score_map = score(ref, hypo)
    print(score_map)