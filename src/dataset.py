from torch.utils.data import IterableDataset, Dataset
import numpy as np
import random
import torch


class DatasetTrain(IterableDataset):
    def __init__(self, filename, news_index, news_combined, args, news_ctr):
        super(DatasetTrain).__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_combined = news_combined
        self.args = args
        self.news_ctr = news_ctr
        # add ctr here as well

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype="float32")

    def line_mapper(self, line):
        line = line.strip().split("\t")
        click_docs = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        click_docs, log_mask = self.pad_to_fix_len(
            self.trans_to_nindex(click_docs), self.args.user_log_length
        )
        user_feature = self.news_combined[click_docs]

        user_feature_ctr = torch.Tensor([self.news_ctr[k] for k in click_docs])

        pos = self.trans_to_nindex(sess_pos)
        neg = self.trans_to_nindex(sess_neg)

        label = random.randint(0, self.args.npratio)
        sample_news = neg[:label] + pos + neg[label:]
        news_feature = self.news_combined[sample_news]

        ##########
        news_feature_ctr = torch.Tensor([self.news_ctr[k] for k in sample_news])
        ########
        # add ctr here as well

        return (
            user_feature,
            log_mask,
            news_feature,
            label,
            news_feature_ctr,
            user_feature_ctr,
        )  # add ctr here as well

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)


class DatasetTest(DatasetTrain):
    def __init__(self, filename, news_index, news_scoring, args, news_ctr):
        super(DatasetTrain).__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_scoring = news_scoring
        self.args = args
        self.news_ctr = news_ctr
        # add ctr here as well

    def line_mapper(self, line):
        line = line.strip().split("\t")
        click_docs = line[3].split()
        click_docs, log_mask = self.pad_to_fix_len(
            self.trans_to_nindex(click_docs), self.args.user_log_length
        )
        user_feature = self.news_scoring[click_docs]
        user_feature_ctr = torch.Tensor([self.news_ctr[k] for k in click_docs])

        # add ctr for candidates here as well
        candidate_news = self.trans_to_nindex(
            [i.split("-")[0] for i in line[4].split()]
        )
        labels = np.array([int(i.split("-")[1]) for i in line[4].split()])
        # add ctr here as well
        news_feature = self.news_scoring[candidate_news]
        news_feature_ctr = torch.Tensor([self.news_ctr[k] for k in candidate_news])

        return (
            user_feature,
            log_mask,
            news_feature,
            labels,
            news_feature_ctr,
            user_feature_ctr,
        )  # add ctr for candidates here as well

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]
