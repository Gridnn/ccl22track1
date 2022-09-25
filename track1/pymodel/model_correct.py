import torch.nn as nn
# from .modeling_bertpinyin import BertModel
from transformers import BertModel


class BERT_Model(nn.Module):

    def __init__(self, pretrained_model, freeze_bert=False):
        super(BERT_Model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.label_ignore_id = 0
        # self.softmax = nn.Softmax(-1)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.label_ignore_id)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            trg_ids=None,
            pinyin_ids=None
    ):
        # print(input_ids[0])
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # pinyin_ids=pinyin_ids
        )
        logits = self.classifier(bert_output[0])
        loss = None
        # print(logits[0].shape)
        # print(trg_ids.shape)
        # print(trg_ids[0])
        if trg_ids is not None:
            loss = self.loss_function(logits.view(-1, self.bert.config.vocab_size), trg_ids.view(-1))
        return loss, logits
