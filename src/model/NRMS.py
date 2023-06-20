import torch
from torch import nn
import torch.nn.functional as F

from .model_utils import AttentionPooling, MultiHeadSelfAttention


class CtrEncoder(nn.Module):
    def __init__(self):
        super(CtrEncoder, self).__init__()

        self.proj = nn.Sequential(nn.Linear(1, 36), nn.LeakyReLU(), nn.Linear(36, 20))

    def forward(self, x):
        """
        Maps float feature to 12-dimensional embedding
        x: batch_size, n_news
        """
        bs = x.size(0)
        # print(x.squeeze().flatten(),x.squeeze().flatten().T  )
        return self.proj(x.squeeze().flatten().unsqueeze(1))


class NewsEncoder(nn.Module):
    def __init__(self, args, embedding_matrix):
        super(NewsEncoder, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.drop_rate = args.drop_rate
        self.dim_per_head = args.news_dim // args.num_attention_heads
        assert args.news_dim == args.num_attention_heads * self.dim_per_head
        self.multi_head_self_attn = MultiHeadSelfAttention(
            args.word_embedding_dim,
            args.num_attention_heads,
            self.dim_per_head,
            self.dim_per_head,
        )
        self.attn = AttentionPooling(args.news_dim, args.news_query_vector_dim)
        self.ctr = CtrEncoder()

    def forward(self, x, ctr_vec, mask=None):
        """
        x: batch_size, word_num
        mask: batch_size, word_num
        ctr_vec: batch_size, 12
        """
        word_vecs = F.dropout(
            self.embedding_matrix(x.long()), p=self.drop_rate, training=self.training
        )
        multihead_text_vecs = self.multi_head_self_attn(
            word_vecs, word_vecs, word_vecs, mask
        )
        multihead_text_vecs = F.dropout(
            multihead_text_vecs, p=self.drop_rate, training=self.training
        )

        news_vec = self.attn(multihead_text_vecs, mask)
        news_vec = torch.cat([news_vec, ctr_vec], dim=1)

        return news_vec


class UserEncoder(nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        self.dim_per_head = (args.news_dim + 20) // args.num_attention_heads
        assert 20 + args.news_dim == args.num_attention_heads * self.dim_per_head
        self.multi_head_self_attn = MultiHeadSelfAttention(
            args.news_dim + 20,
            args.num_attention_heads,
            self.dim_per_head,
            self.dim_per_head,
        )
        self.attn = AttentionPooling(args.news_dim + 20, args.user_query_vector_dim)
        self.pad_doc = nn.Parameter(
            torch.empty(1, args.news_dim + 20).uniform_(-1, 1)
        ).type(torch.FloatTensor)

    def forward(self, news_vecs, log_mask=None):
        """
        news_vecs: batch_size, history_num, news_dim
        log_mask: batch_size, history_num
        """
        bz = news_vecs.shape[0]
        if self.args.user_log_mask:
            news_vecs = self.multi_head_self_attn(
                news_vecs, news_vecs, news_vecs, log_mask
            )  # news_vecs should include CTR
            user_vec = self.attn(news_vecs, log_mask)
        else:
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(
                bz, self.args.user_log_length, -1
            )  # news_vecs should include CTR
            news_vecs = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (
                1 - log_mask.unsqueeze(dim=-1)
            )
            news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs)
            user_vec = self.attn(news_vecs)
        return user_vec


class Model(torch.nn.Module):
    def __init__(self, args, embedding_matrix, **kwargs):
        super(Model, self).__init__()
        self.args = args
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(
            pretrained_word_embedding, freeze=args.freeze_embedding, padding_idx=0
        )

        self.news_encoder = NewsEncoder(args, word_embedding)
        self.user_encoder = UserEncoder(args)
        self.ctr_encoder = CtrEncoder()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        history,
        history_mask,
        candidate,
        label,
        user_feature_ctr,
        news_feature_ctr,
    ):  # add CTR arguments here
        """
        history: batch_size, history_length, num_word_title
        history_mask: batch_size, history_length
        candidate: batch_size, 1+K, num_word_title
        label: batch_size, 1+K
        user_feature_ctr: batch_size, history_length
        news_feature_ctr: batch_size, 1+K
        """
        assert news_feature_ctr.size(1) == candidate.size(1)
        assert user_feature_ctr.size(1) == history.size(1)
        candidate_news = candidate.reshape(-1, self.args.num_words_title)
        ctr_vec2 = self.ctr_encoder(news_feature_ctr)
        candidate_news_vecs = self.news_encoder(candidate_news, ctr_vec2).reshape(
            -1, 1 + self.args.npratio, self.args.news_dim + 20
        )
        # add CTR to candidate_news_vecs here

        ctr_vec1 = self.ctr_encoder(user_feature_ctr)
        history_news = history.reshape(-1, self.args.num_words_title)
        history_news_vecs = self.news_encoder(history_news, ctr_vec1).reshape(
            -1, self.args.user_log_length, self.args.news_dim + 20
        )
        # add CTR to history_news_vecs here

        user_vec = self.user_encoder(history_news_vecs, history_mask)
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(
            dim=-1
        )
        loss = self.loss_fn(score, label)
        return loss, score
