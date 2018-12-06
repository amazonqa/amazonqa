"""
Computes the BLEU, ROUGE
using the COCO metrics scripts
"""
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

class AmazonQAEvaulator:

    def compute_scores(ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(),"METEOR"),
            (Cider(), "CIDEr")
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
    
def compute_metrics_from_files(path_to_reference_file, path_to_candidate_file):
    pass

def main():
    path_to_reference_file = sys.argv[1]
    path_to_candidate_file = sys.argv[2]

    metrics = compute_metrics_from_files(path_to_reference_file, path_to_candidate_file)
    print('############################')
    for metric in sorted(metrics):
        print('%s: %.4f' % (metric, metrics[metric]))
    print('############################')

if __name__ == "__main__":
    main()
