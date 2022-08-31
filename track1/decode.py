import torch
from transformers import BertTokenizer
from utils import *
from csc.csc_model import CSCModel
from tqdm import tqdm
import json
import argparse


class Decoder:
    def __init__(self, config):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.test_loader = init_dataloader(config.test_path, config, "test", self.tokenizer)
        self.model = CSCModel(config.pretrained_model)
        self.model.to(self.device)
        self.config = config

    def __forward_prop(self, dataloader, back_prop=False):
        collected_outputs = []
        prob_outputs = []
        final_out = []
        softmax = torch.nn.Softmax(-1)
        f = open(self.config.save_path + '.txt', "w")
        for id, batch in tqdm(enumerate(dataloader)):
            f.write(str(id + 1) + ':' + '\n')
            batch = {k: v.to(self.device) for k, v in batch.items()}
            for repeat in range(5):
                loss, logits = self.model(input_ids=batch['input_ids'],
                                          attention_mask=batch['attention_mask'],
                                          token_type_ids=batch['token_type_ids'],
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
        return final_out, prob_outputs

    def __forward_prop_all(self, dataloader, back_prop=False):
        collected_outputs = []
        prob_outputs = []
        softmax = torch.nn.Softmax(-1)
        for batch in tqdm(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss, logits = self.model(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      token_type_ids=batch['token_type_ids'],
                                      )

            outputs = torch.argmax(logits, dim=-1).cpu()
            outputs_prob = torch.max(softmax(logits), dim=-1)[0].cpu()

            for outputs_i, prob_i in zip(outputs, outputs_prob):
                collected_outputs.append(outputs_i)
                prob_outputs.append(prob_i)

        return collected_outputs, prob_outputs

    def save_as_json(self, collected_outputs, prob_outputs, data, path):
        result = {}
        path += '.json'
        for index, (pred_i, prob_i, src) in enumerate(zip(collected_outputs, prob_outputs, data)):
            src_i = src['input_ids']
            line = ""
            pred_i = pred_i[:len(src_i)]
            pred_i = pred_i[1:-1]

            prob_i = prob_i[:len(src_i)]
            prob_i = prob_i[1:-1]

            src_i = src_i[1:-1]
            proba = []
            for id, (ele, prob, trg) in enumerate(zip(pred_i, prob_i, src_i)):
                # ['很', '和', '少', '小']
                ele != src
                if ele != src_i[id] and vocab[ele] != "[UNK]" and is_all_chinese(vocab[ele]) and prob>0.5:
                    line += vocab[ele]
                    proba.append(prob.item())
                else:
                    line += src['src_text'][id]
                    proba.append(0)
            sample_id = '(YACLC-CSC-TEST-ID=' + "{:0>4d}".format(index + 1) + ')'
            # sample_id = "hybrid_" + str(index)
            content = {
                "text": src['src_text'],
                "correction": line,
                "proba": proba
            }
            result.update({sample_id: content})
        with open(path, 'w') as f:
            json.dump(result, f, ensure_ascii=False)

    def decode(self):
        model = self.model
        model.load_state_dict(torch.load(self.config.model_path))
        model.eval()
        with torch.no_grad():
            test_output, test_prob = self.__forward_prop(dataloader=self.test_loader, back_prop=False)
            save_decode_result_lbl(test_output, test_prob, self.test_loader.dataset.data, self.config.save_path)
            # save_decode_result_para(outputs, self.test_loader.dataset.data, self.config.save_path)
            self.save_as_json(test_output, test_prob, self.test_loader.dataset.data, self.config.save_path)


def main(config):
    decoder = Decoder(config)
    decoder.decode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--label_ignore_id", default=0, type=int)

    args = parser.parse_args()
    main(args)
