from dparser.metrics import AttachmentMethod

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence


class Model():
    def __init__(self, config, vocab, parser):

        self.config = config
        self.vocab = vocab
        self.parser = parser

    def train(self, loader):
        self.parser.train()
        with torch.autograd.detect_anomaly():
            for i, (words, chars, tags, arcs, rels) in enumerate(loader):
                mask = words.ne(self.vocab.pad_index)
                # ignore the first token of each sentence
                mask[:, 0] = 0
                s_arc, s_rel = self.parser(words, chars)
                s_arc, s_rel = s_arc[mask], s_rel[mask]
                gold_arcs, gold_rels = arcs[mask], rels[mask]

                loss = self.parser.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
                loss = loss / self.config.update_steps
                loss.backward()

                if (i + 1) % self.config.update_steps == 0:
                    # nn.utils.clip_grad_norm_(self.parser.parameters(),
                    #                          self.config.clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, loader, punct=True, partial=False, tree=False):
        self.parser.eval()

        loss, metirc = 0, AttachmentMethod()

        for words, chars, tags, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            mask[:, 0] = 0

            s_arc, s_rel = self.parser(words, chars)
            pred_arcs, pred_rels = self.parser.decode(
                s_arc, s_rel, mask, tree=tree)

            if partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if specified
            if not punct:
                puncts = words.new_tensor(self.vocab.puncts)
                mask &= words.unsqueeze(-1).ne(puncts).all(-1)

            s_arc, s_rel = s_arc[mask], s_rel[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]
            pred_arcs, pred_rels = pred_arcs[mask], pred_rels[mask]

            loss += self.parser.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            metirc(pred_arcs, pred_rels, gold_arcs, gold_rels)
        loss /= len(loader)

        return loss, metirc

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()

        all_arcs, all_rels = [], []
        for words, chars, tags in loader:
            mask = words.ne(self.vocab.pad_index)
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_arc, s_rel = self.parser(words, chars)
            pred_arcs, pred_rels = self.parser.decode(s_arc, s_rel, mask, True)
            all_arcs.extend(torch.split(pred_arcs[mask], lens))
            all_rels.extend(torch.split(pred_rels[mask], lens))

            # s_arc, s_rel = s_arc[mask], s_rel[mask]
            # pred_arcs, pred_rels = self.parser.decode(s_arc, s_rel, mask)

            # all_arcs.extend(torch.split(pred_arcs, lens))
            # all_rels.extend(torch.split(pred_rels, lens))
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_arcs, all_rels

    # @torch.no_grad()
    # def evaluate(self, loader, punct=True, partial=False):
    #     self.parser.eval()

    #     loss, metirc = 0, AttachmentMethod()

    #     for words, chars, tags, arcs, rels in loader:
    #         mask = words.ne(self.vocab.pad_index)
    #         mask[:, 0] = 0

    #         s_arc, s_rel = self.parser(words, chars)
    #         pred_arcs, pred_rels, *_ = self.parser.decode(s_arc, s_rel, mask)

    #         if partial:
    #             mask &= arcs.ge(0)
    #         # ignore all punctuation if specified
    #         if not punct:
    #             puncts = words.new_tensor(self.vocab.puncts)
    #             mask &= words.unsqueeze(-1).ne(puncts).all(-1)

    #         s_arc, s_rel = s_arc[mask], s_rel[mask]
    #         gold_arcs, gold_rels = arcs[mask], rels[mask]
    #         pred_arcs, pred_rels = pred_arcs[mask], pred_rels[mask]

    #         loss += self.parser.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
    #         metirc(pred_arcs, pred_rels, gold_arcs, gold_rels)
    #     loss /= len(loader)

    #     return loss, metirc

    # @torch.no_grad()
    # def predict(self, loader):
    #     self.parser.eval()

    #     all_arcs, all_rels = [], []
    #     scores = []
    #     for words, chars, tags in loader:
    #         mask = words.ne(self.vocab.pad_index)
    #         mask[:, 0] = 0
    #         lens = mask.sum(dim=1).tolist()
    #         s_arc, s_rel = self.parser(words, chars)

    #         # s_arc, s_rel = s_arc[mask], s_rel[mask]
    #         # pred_arcs, pred_rels = self.parser.decode(s_arc, s_rel, mask, True)

    #         # all_arcs.extend(torch.split(pred_arcs[mask], lens))
    #         # all_rels.extend(torch.split(pred_rels[mask], lens))
    #         pred_arcs, pred_rels, pred_arcs_score, pred_rels_score = self.parser.decode(
    #             s_arc, s_rel, mask)

    #         # pred_arcs_scores = pad_sequence(torch.split(
    #         #     pred_arcs_score[mask], lens), True).sum(dim=-1) / mask.sum(dim=1)

    #         pred_rels_scores = pad_sequence(torch.split(
    #             pred_rels_score[mask], lens), True).sum(dim=-1) / mask.sum(dim=1)

    #         all_arcs.extend(torch.split(pred_arcs[mask], lens))
    #         all_rels.extend(torch.split(pred_rels[mask], lens))
    #         scores.extend(pred_rels_scores)
    #     all_arcs = [seq.tolist() for seq in all_arcs]
    #     all_rels = [self.vocab.id2rel(seq) for seq in all_rels]
    #     sorted_scores, indices = torch.sort(torch.tensor(scores), descending=True)

    #     return all_arcs, all_rels, sorted_scores, indices
