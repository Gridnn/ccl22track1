from utils import *
from transformers import BertTokenizer, AdamW, get_scheduler
from tqdm import tqdm
from pymodel.model import BERT_Model
import torch
import os
import argparse
from random import seed


class Trainer:

    def __init__(self, config):
        self.config = config
        self.fix_seed(config.seed)
        print(config.__dict__)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.train_dataloader = init_dataloader(config.train_path, config, "train", self.tokenizer)
        self.valid_dataloader = init_dataloader(config.dev_path, config, "dev", self.tokenizer)
        for i in self.valid_dataloader.dataset.data:
            print(i)
            break
        self.test_dataloader = init_dataloader(config.test_path, config, "testtt", self.tokenizer)
        self.model = BERT_Model(config)
        # self.model = SoftMaskedBert4Csc(cfg=config, device=self.device, tokenizer=self.tokenizer)
        # self.model.load_state_dict(torch.load('./model/sighan_dedide.pt'))
        # print("model is loaded")
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = self.set_scheduler()
        self.best_score = {"valid-c": 0, "valid-s": 0}
        self.best_epoch = {"valid-c": 0, "valid-s": 0}

    def fix_seed(self, seed_num):
        torch.manual_seed(seed_num)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed(seed_num)

    def set_scheduler(self):
        num_epochs = self.config.num_epochs
        num_training_steps = num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        return lr_scheduler

    def __forward_prop(self, dataloader, back_prop=False):
        collected_outputs = []
        prob_outputs = []
        final_out = []
        softmax = torch.nn.Softmax(-1)
        f = open(self.config.save_path + '.txt', "w")
        for id, batch in tqdm(enumerate(dataloader)):
            f.write(str(id + 1) + ':' + '\n')
            batch = {k: v.to(self.device) for k, v in batch.items()}
            for repeat in range(10):
                loss, logits = self.model(input_ids=batch['input_ids'],
                                          attention_mask=batch['attention_mask'],
                                          token_type_ids=batch['token_type_ids'],
                                          pinyin_ids=batch['pinyin_ids']
                                          )

                outputs = torch.argmax(logits, dim=-1).cpu()
                outputs_prob = torch.max(softmax(logits), dim=-1)[0].cpu()
                outputs_prob_k, outputs_index_k = torch.topk(softmax(logits), k=3, dim=-1)
                # for outputs_i, prob_i in zip(outputs, outputs_prob):
                #     collected_outputs.append(outputs_i)
                #     prob_outputs.append(prob_i)

                for prob_i in outputs_prob:
                    prob_outputs.append(prob_i)
                for outputs_line, prob_line, batch_line, prob_k, index_k in zip(outputs, outputs_prob,
                                                                                batch['input_ids'], outputs_prob_k,
                                                                                outputs_index_k):
                    max_prob = 0
                    threshold = 0.0
                    for n, (outputs_i, prob_i, batch_i, prob_k_i, index_k_i) in enumerate(
                            zip(outputs_line, prob_line, batch_line, prob_k, index_k)):
                        outputs_i = outputs_i.item()
                        prob_i = prob_i.item()
                        batch_i = batch_i.item()
                        prob_k_i = prob_k_i
                        index_k_i = index_k_i
                        if outputs_i != batch_i and prob_i > max_prob and batch_i not in [0, 101, 102]:
                            max_prob = prob_i
                            max_index = n
                            max_output = outputs_i

                            second_prob, third_prob = prob_k_i[1].item(), prob_k_i[2].item()
                            second_output, third_output = index_k_i[1].item(), index_k_i[2].item()
                    if max_prob > threshold:
                        origin = batch_line[max_index].clone()
                        batch_line[max_index] = max_output
                        f.write('repeat:' +
                                str(repeat) + ', ' +
                                str(max_index) + ', ' +
                                self.tokenizer.decode(origin) + 'to' +
                                self.tokenizer.decode(max_output) + ' ' + str(max_prob) + ' ' +
                                self.tokenizer.decode(second_output) + ' ' + str(second_prob) + ' ' +
                                self.tokenizer.decode(third_output) + ' ' + str(third_prob) +
                                '\n')
            f.write('\n')
            for line in batch['input_ids']:
                final_out.append([i.item() for i in line])
        return None, final_out, prob_outputs

    def __forward_prop_all(self, dataloader, back_prop=True):
        loss_sum = 0
        steps = 0
        collected_outputs = []
        prob_outputs = []
        softmax = torch.nn.Softmax(-1)
        for batch in tqdm(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss, logits = self.model(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      token_type_ids=batch['token_type_ids'],
                                      trg_ids=batch['trg_ids'],
                                      pinyin_ids=batch['pinyin_ids'])

            # det_loss, cor_loss, prob, logits, sequence_output = self.model(input_ids=batch['input_ids'],
            #                                                                attention_mask=batch['attention_mask'],
            #                                                                cor_labels=batch['trg_ids'],
            #                                                                )
            # loss = det_loss + cor_loss
            loss_sum += loss.item()
            if back_prop:
                loss.backward()
                # 对抗训练
                loss_adv, _ = self.model(input_ids=batch['input_ids'],
                                         attention_mask=batch['attention_mask'],
                                         token_type_ids=batch['token_type_ids'],
                                         trg_ids=batch['trg_ids'],
                                         pinyin_ids=batch['pinyin_ids'])
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            else:
                outputs = torch.argmax(logits, dim=-1).cpu()
                outputs_prob = torch.max(softmax(logits), dim=-1)[0].cpu()

                for outputs_i, prob_i in zip(outputs, outputs_prob):
                    collected_outputs.append(outputs_i)
                    prob_outputs.append(prob_i)
            steps += 1
        epoch_loss = loss_sum / steps
        return epoch_loss, collected_outputs, prob_outputs

    def __save_ckpt(self, epoch):
        save_path = self.config.save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        path = os.path.join(save_path, self.config.tag + str(epoch) + ".pt")
        torch.save(self.model.state_dict(), path)

    def train(self):
        no_improve = 0
        most_count = 0
        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            train_loss, _, _ = self.__forward_prop_all(self.train_dataloader, back_prop=True)
            self.model.eval()
            with torch.no_grad():
                valid_loss, valid_output, valid_prob = self.__forward_prop(self.valid_dataloader, back_prop=False)
            # print(f"train_loss: {train_loss}, valid_loss: {valid_loss}")
            if not os.path.exists(self.config.save_path + '/tmp/'):
                os.makedirs(self.config.save_path + '/tmp/')
            save_decode_result_para(valid_output, valid_prob, self.valid_dataloader.dataset.data,
                                    self.config.save_path + '/tmp/' + "valid_" + str(epoch) + ".txt")
            count = save_decode_result_lbl(valid_output, self.valid_dataloader.dataset.data,
                                           self.config.save_path + '/tmp/' + "valid_" + str(epoch) + ".lbl")
            # try:
            char_metrics, sent_metrics = csc_metrics(
                pred=self.config.save_path + '/tmp/' + "valid_" + str(epoch) + ".lbl",
                gold=self.config.lbl_path,
                src='data/yaclc-csc_dev.src',
            )
            get_best_score(self.best_score, self.best_epoch, epoch,
                           char_metrics["Correction"]["F1"], sent_metrics["Correction"]["F1"])
            if max(self.best_epoch.values()) == epoch:
                self.__save_ckpt(epoch)

            print(f"curr epoch: {epoch} | curr best epoch {self.best_epoch}")
            print(f"best socre:{self.best_score}")
            print(f"no improve: {epoch - max(self.best_epoch.values())}")
            if (epoch - max(self.best_epoch.values())) >= self.config.patience:
                break


def main(config):
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", required=True, type=str)
    parser.add_argument("--train_path", required=True, type=str)
    parser.add_argument("--dev_path", required=True, type=str)
    parser.add_argument("--test_path", required=True, type=str)
    parser.add_argument("--lbl_path", required=True, type=str)
    parser.add_argument("--test_lbl_path", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--label_ignore_id", default=0, type=int)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--freeze_bert", default=False, type=bool)
    parser.add_argument("--tie_cls_weight", default=False, type=bool)
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--seed", default=2021, type=int)
    args = parser.parse_args()
    main(args)
