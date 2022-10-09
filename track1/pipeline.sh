# Step 1. Data preprocessing
DATA_DIR=./exp_data/sighan
PRETRAIN_MODEL=./csc/bert
mkdir -p $DATA_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3


TRAIN_SRC_FILE=data/yaclc-csc_dev.src
TRAIN_TRG_FILE=data/yaclc-csc_dev.trg
DEV_SRC_FILE=data/yaclc-csc_dev.src
DEV_TRG_FILE=data/yaclc-csc_dev.lbl
TEST_SRC_FILE=data/yaclc-csc_dev.src
TEST_TRG_FILE=data/yaclc-csc_dev.lbl

if [ ! -f $DATA_DIR"/train.pkl" ]; then
    python ./data_preprocess.py \
    --source_dir $TRAIN_SRC_FILE \
    --target_dir $TRAIN_TRG_FILE \
    --bert_path $PRETRAIN_MODEL \
    --save_path $DATA_DIR"/train.pkl" \
    --data_mode "para" \
    --normalize "True"
fi

if [ ! -f $DATA_DIR"/dev.pkl" ]; then
    python ./data_preprocess.py \
    --source_dir $DEV_SRC_FILE \
    --target_dir $DEV_TRG_FILE \
    --bert_path $PRETRAIN_MODEL \
    --save_path $DATA_DIR"/dev.pkl" \
    --data_mode "lbl" \
    --normalize "True"
fi

if [ ! -f $DATA_DIR"/test.pkl" ]; then
    python ./data_preprocess.py \
    --source_dir $TEST_SRC_FILE \
    --target_dir $TEST_TRG_FILE \
    --bert_path $PRETRAIN_MODEL \
    --save_path $DATA_DIR"/test.pkl" \
    --data_mode "lbl" \
    --normalize "True"
fi


# Step 2. Training
MODEL_DIR=./exps/sighan_new
CUDA_DEVICE=0
mkdir -p $MODEL_DIR/bak
cp ./pipeline.sh $MODEL_DIR/bak

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train_pipeline.py \
    --pretrained_model $PRETRAIN_MODEL \
    --train_path $DATA_DIR"/train.pkl" \
    --dev_path $DATA_DIR"/dev.pkl" \
    --test_path $DATA_DIR"/test.pkl" \
    --lbl_path $DEV_TRG_FILE \
    --test_lbl_path $TEST_TRG_FILE \
    --save_path $MODEL_DIR \
    --batch_size 32 \
    --num_epochs 10 \
    --lr 5e-5 \
    --tie_cls_weight False \
    --tag "sighan" \
    2>&1 | tee $MODEL_DIR"/log.txt"


# prediction
