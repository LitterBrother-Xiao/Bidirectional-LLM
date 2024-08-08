# xsum
#!/bin/bash
# dir=`pwd`
###############################################################################
### Main configs
### The main configs are from Megatron-LM paper
### https://arxiv.org/abs/1909.08053. Choose based on your desired model size
### or build your own configs.
model_size=0.11
num_layers=12
hidden_size=768
num_attn_heads=12
init_std=0.02

## 336M (same config as original BERT-Large model)
# model_size=0.336
# num_layers=24
# hidden_size=1024
# num_attn_heads=16

## 1.3B
# model_size=1.3
# num_layers=24
# hidden_size=2048
# num_attn_heads=32
# init_std=0.013

## 3.9B
# model_size=3.9
# num_layers=48
# hidden_size=2560
# num_attn_heads=40
# init_std=0.011
###############################################################################

### Training duration configs
## The main termination condition, original Megatron paper trains for 2M iters.

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Currently pipeline parallelism is not supported for BERT model: DeepSpeed's
## pipeline parallelism is only integrated with the GPT case, and currently
## DeepSpeed is not integrated with Megatron's own pipeline parallelism.
pp_size=1
no_pp="true"

TRAIN_DATA_PATH="xxx"
VALID_DATA_PATH="xxx"
TOKENIZER_PATH=xxx

output_home="xxx"
log_path="${output_home}/log/"
checkpoint_path="${output_home}/xxxx"
pretraining_model_path="${output_home}/xxxx"
## Microsoft internal constraint: because tensorboard is logged by last rank,
## it's better to put the path in NFS instead of Blob.
mkdir -p ${checkpoint_path}
mkdir -p ${log_path}
###############################################################################
data_options=" \
    --train-data-path $TRAIN_DATA_PATH \
    --valid-data-path $VALID_DATA_PATH \
    --tokenizer-model $TOKENIZER_PATH \
    --tokenizer-type HFTokenizer \
    --data-impl mmap "

global_batch_size=8
batch_size=1


megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size 1 \
    --lr-decay-iters 200000 \
    --lr-warmup-iters 2000 \
    --micro-batch-size ${batch_size} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters 200000 \
    --lr 1e-4 \
    --min-lr 1e-7 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-interval 50000 \
    --eval-iters 1 \
    --save-interval 2429 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --load ${pretraining_model_path} \
    --save ${checkpoint_path} \
    --finetune \
    --swiglu \
    --use-rotary-position-embeddings \
    --rotary-percent 0.25 \
    --num-workers 16 \
    --no-load-rng \
    --seed 6666 "

###### if length-prediction
# --length-predict
# --max-predict-length
# --length-factor

###### if dpo training
#--dpo-trainingß
#--dpo-update-model-step
#-dpo-sampling-type
#--dpo-type


log_interval=100
zero_stage=1
config_json="../ds_config/ds_config_xsum.json"
deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --pipeline-model-parallel-size ${pp_size}"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29000  ../../pretrain_gebert.py ${megatron_options} ${data_options} ${deepspeed_options} &> ${log_path}/xxx.log
