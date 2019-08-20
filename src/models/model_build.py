import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from models.encoder import Classifier, DNNEncoder


class Bert(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()
        if load_pretrained_bert:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel(bert_config)

    def forward(self, x, segs):
        encoded_layers, _ = self.model(x, segs, attention_mask=None)
        top_vec = encoded_layers[-1]
        return top_vec


class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert=False, bert_config=None):
        super(Summarizer, self).__init__()
        self.device = device
        self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        if args.encoder == "classifier":
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        if args.encoder == "dnn":
            self.encoder = DNNEncoder(self.bert.model.config.hidden_size, args.num_units, args.num_layers)
        self.to(device)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss):
        top_vec = self.bert(x, segs)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sent_scores = self.encoder(sents_vec).squeeze(-1)
        return sent_scores
