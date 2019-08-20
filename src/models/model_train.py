import numpy as np
import torch

from prepro.utils import tidy


class Trainer(object):

    def __init__(self, args, device, model):
        self.args = args
        self.block_tri = args.block_trigram
        self.block_low = args.block_lower
        self.model = model
        self.device = device

    def infer(self, example, num_sentence=3):
        """
        Inference function for BertSum
        """

        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        def _block_low(array, score):
            mean = np.mean(array)
            return score < mean

        self.model.eval()

        src = example['src']
        segs = example['segs']
        clss = example['clss']
        src_str = example['src_str']

        device = self.device
        result_path = '%s/summary.text' % self.args.save_path

        if len(src_str) == 0:
            print("\n[Error] The text has zero input ...")
            return

        if len(src_str) == 1:
            with open(result_path, 'w') as summary:
                candidate = src_str[0]
                candidate = tidy(candidate)
                summary.write(candidate)
                print(candidate)
            return

        src = torch.tensor(src).unsqueeze(0).to(device)
        segs = torch.tensor(segs).unsqueeze(0).to(device)
        clss = torch.tensor(clss).unsqueeze(0).to(device)

        pred = []

        sent_scores = self.model(src, segs, clss)
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)
        index = 0

        with open(result_path, 'w') as summary:
            for j in selected_ids[0]:
                candidate = src_str[j].strip()
                candidate = tidy(candidate)
                candidate_score = sent_scores[0][j]

                if not(self.block_tri and _block_tri(candidate, pred)):
                    if not (self.block_low and _block_low(sent_scores[0], candidate_score)):
                        pred.append(candidate)
                        summary.write(candidate + "\n")
                        index += 1

                if len(pred) >= num_sentence:
                    break

        pred = ' '.join(pred)

        print("\n[INFO] This is the summary:\n")
        print(pred)
        print("\n[INFO] The summary produced %d sentences" % index)
