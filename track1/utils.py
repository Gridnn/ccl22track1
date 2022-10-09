from torch.utils.data import DataLoader
from dataset import CSC_Dataset, Padding_in_batch
from eval_char_level import get_char_metrics
from eval_sent_level import get_sent_metrics

vocab_path = "csc/bert/vocab.txt"
vocab = []
with open(vocab_path, "r") as f:
    lines = f.readlines()
for line in lines:
    vocab.append(line.strip())


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def init_dataloader(path, config, subset, tokenizer):
    sub_dataset = CSC_Dataset(path, config, subset)

    if subset == "train":
        is_shuffle = True
    else:
        is_shuffle = False

    collate_fn = Padding_in_batch(0)

    data_loader = DataLoader(
        sub_dataset,
        batch_size=config.batch_size,
        shuffle=is_shuffle,
        collate_fn=collate_fn
    )

    return data_loader


def csc_metrics(pred, gold, src='../../datasets/track1/test/yaclc-csc_test.src'):
    print(src)
    char_metrics = get_char_metrics(src_path=src, pred_path=pred, gold_path=gold)
    sent_metrics = get_sent_metrics(pred_path=pred, targ_path=gold)
    return char_metrics, sent_metrics


def get_best_score(best_score, best_epoch, epoch, *params):
    for para, key in zip(params, best_score.keys()):
        if para > best_score[key]:
            best_score[key] = para
            best_epoch[key] = epoch
    return best_score, best_epoch


def save_decode_result_para(decode_pred, prob_pred, data, path, threshold=0.9):
    # print(decode_pred.shape)
    # print(prob_pred)
    # print(prob_pred.shape)
    f = open(path, "w")
    for i, (pred_i, prob_i, src) in enumerate(zip(decode_pred, prob_pred, data)):
        src_i = src['input_ids']
        line = ""
        pred_i = pred_i[:len(src_i)]
        pred_i = pred_i[1:-1]

        prob_i = prob_i[:len(src_i)]
        prob_i = prob_i[1:-1]
        for id, (ele, prob) in enumerate(zip(pred_i, prob_i)):
            if vocab[ele] != "[UNK]" and is_all_chinese(vocab[ele]) and prob > threshold:
                line += vocab[ele]
            else:
                line += src['src_text'][id]
        f.write("input:" + src['src_text'] + "\n")
        f.write("inference:" + line + "\n")
        f.write("trg:" + src['trg_text'] + "\n\n")
    f.close()
    # f = open(path, "w")
    # for i, (pred_i, src) in enumerate(zip(decode_pred, data)):
    #     src_i = src['input_ids']
    #     line = ""
    #     pred_i = pred_i[:len(src_i)]
    #     pred_i = pred_i[1:-1]
    #     for id, ele in enumerate(pred_i):
    #         if ele == 1:
    #             line += '[MASK]'
    #         else:
    #             line += src['src_text'][id]
    #     f.write("input:" + src['src_text'] + "\n")
    #     f.write("inference:" + line + "\n")
    #     f.write("trg:" + src['trg_text'] + "\n\n")
    # f.close()


def save_decode_result_lbl(decode_pred, data, path):
    count = 0
    with open(path, "w") as fout:
        for pred_i, src in zip(decode_pred, data):
            src_i = src['input_ids']
            line = src['id'] + ", "
            pred_i = pred_i[:len(src_i)]
            no_error = True

            for id, ele in enumerate(pred_i):
                if id != 0 and id != len(pred_i) - 1:
                    if ele != src_i[id] and is_all_chinese(vocab[ele]) and is_all_chinese(vocab[src_i[id]]):
                            if vocab[ele] != "[UNK]":
                                count += 1
                                no_error = False
                                line += (str(id) + ", " + vocab[ele] + ", ")
            if no_error:
                line += '0'
            line = line.strip(", ")
            fout.write(line + "\n")
    print("Number of detected errors", count)
    return count
# with open(path, "w") as fout:
#     for pred_i, src in zip(decode_pred, data):
#         src_i = src['input_ids']
#         line = src['id'] + ", "
#         pred_i = pred_i[:len(src_i)]
#         no_error = True
#         for id, ele in enumerate(pred_i):
#             if id != 0 and id != len(pred_i):
#                 if ele == 1:
#                     no_error = False
#                     line += (str(id) + ", " + "[MASK]" + ", ")
#         if no_error:
#             line += '0'
#         line = line.strip(", ")
#         fout.write(line + "\n")


def save_decode_result_mask(decode_pred, data, path):
    path += '.mask'
    f = open(path, "w")
    for i, (pred_i, src) in enumerate(zip(decode_pred, data)):
        src_i = src['input_ids']
        line = src['id'] + "\t"
        pred_i = pred_i[:len(src_i)]
        pred_i = pred_i[1:-1]
        for id, ele in enumerate(pred_i):
            if ele == 1:
                line += '[MASK]'
            else:
                line += src['src_text'][id]
        f.write(line + "\n")
    f.close()