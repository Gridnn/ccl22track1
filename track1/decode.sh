PRETRAIN_MODEL=./csc/bert
DATA_DIR=./exp_data/sighan
# hybird_test.src
TEST_SRC_FILE=../../datasets/track1/test/single.txt
TAG=yaclc-csc-test_dedide

python ./data_preprocess.py \
--source_dir $TEST_SRC_FILE \
--bert_path $PRETRAIN_MODEL \
--save_path $DATA_DIR"/test_"$TAG".pkl" \
--data_mode "lbl" \
--normalize "True"

MODEL_PATH=model/train_with_dev.pt
SAVE_PATH=exps/sighan/decode

mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=0 python decode.py \
    --pretrained_model $PRETRAIN_MODEL \
    --test_path $DATA_DIR"/test_"$TAG".pkl" \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH"/"$TAG".lbl" ;

