PRETRAIN_MODEL=csc/bert
DATA_DIR=data
TEST_SRC_FILE=data/yaclc-csc_test.src
TAG=yaclc-csc-test

python ./data_preprocess.py \
--source_dir $TEST_SRC_FILE \
--bert_path $PRETRAIN_MODEL \
--save_path $DATA_DIR"/test_"$TAG".pkl" \
--data_mode "lbl" \
--normalize "True"

MODEL_PATH=csc/bert/final_model.pt
SAVE_PATH=data/decode

mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=0 python decode.py \
    --pretrained_model $PRETRAIN_MODEL \
    --test_path $DATA_DIR"/test_"$TAG".pkl" \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH"/"$TAG".lbl" ;


python fluent/fluent.py