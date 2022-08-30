import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from transformers import BertModel
import math
from torch.nn import Parameter


class BERT_CLS(nn.Module):
    def __init__(self, args, tag2idx):
        super(BERT_CLS, self).__init__()
        self.bert_model = BertModel.from_pretrained(args.bert_path)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, len(tag2idx))
        
    def forward(self, sentence):
        input_mask = (sentence!=0)
        embed = self.bert_model(sentence, attention_mask=input_mask, token_type_ids=None)
        embed = embed["last_hidden_state"][:, 0,:]
        embed = self.dropout(embed)
        output = self.linear(embed)
        return output


class BERT_POOLING(nn.Module):
    def __init__(self, args, tag2idx):
        super(BERT_POOLING, self).__init__()
        self.bert_model = BertModel.from_pretrained(args.bert_path)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, len(tag2idx))


    def forward(self, sentence):
        input_mask = (sentence!=0)
        embed = self.bert_model(sentence, attention_mask=input_mask, token_type_ids=None)["last_hidden_state"][:, 1: -1, :]
        # POOLING without PADDING
        input_mask_expanded = input_mask[:, 1: -1].unsqueeze(-1).expand(embed.size()).float()
        sum_embeddings = torch.sum(embed * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        sum_embeddings = (sum_embeddings / sum_mask)
        embed = self.dropout(sum_embeddings)
        output = self.linear(embed)
        return output, sum_embeddings


# 以下のコードはhttps://github.com/4uiiurz1/pytorch-adacosより引用しました。
class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)


    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.training:
            with torch.no_grad():
                B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
                B_avg = torch.sum(B_avg) / input.size(0)
                # print(B_avg)
                theta_med = torch.median(theta[one_hot == 1])
                self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits
        return output