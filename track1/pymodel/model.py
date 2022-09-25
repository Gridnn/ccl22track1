import torch.nn as nn
from .modeling_bertpinyin import BertModel, BertForMaskedLM


class BERT_Model(nn.Module):

    def __init__(self, config, freeze_bert=False):
        super(BERT_Model, self).__init__()

        print(config.pretrained_model)
        self.bert = BertForMaskedLM.from_pretrained(config.pretrained_model)

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
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=trg_ids,
            pinyin_ids=pinyin_ids
        )
        # logits = self.classifier(bert_output[0])
        # loss = None
        # if trg_ids is not None:
        #     loss = self.loss_function(logits.view(-1, self.bert.config.vocab_size), trg_ids.view(-1))
        if trg_ids is None:
            loss = None
            logits = bert_output[0]
        else:
            loss, logits = bert_output[0], bert_output[1]
        return loss, logits

    # def tie_cls_weight(self):
    #     self.classifier.weight = self.bert.embeddings.word_embeddings.weight


class BERT_Model_pinyin(nn.Module):

    def __init__(self, config, freeze_bert=False, tie_cls_weight=True):
        super(BERT_Model_pinyin, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.label_ignore_id = 2
        self.softmax = nn.Softmax(-1)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.label_ignore_id)
        # if tie_cls_weight:
        #     self.tie_cls_weight()

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
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pinyin_ids=pinyin_ids
        )
        # print(bert_output)
        logits = self.classifier(bert_output[0])
        loss = None
        if trg_ids is not None:
            loss = self.loss_function(logits.view(-1, 2), trg_ids.view(-1))
        return loss, logits
