# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=17099
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT  
"

seq_len=2048
max_posi_emb=2048
global_batch_size=1024
batch_size=4
lr=6e-4
min_lr=6e-5


model_size=0.11
num_layers=12
hidden_size=768
num_attn_heads=12
init_std=0.02
num_kv_nums=12


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
train_iters=153000
lr_warmup_iters=1530
lr_decay_iters=153000
lr_decay_style="cosine"
###############################################################################
### Parallelism configs
## Model parallelism, 1 is no MP
mp_size=1

pp_size=1
no_pp="true"

## ZeRO stage
zero_stage=1

## Total number of GPUs. ds_ssh is from DeepSpeed library.
# num_gpus=1
# num_gpus_pernode=1
# num_node=1
# # ## Data parallel size.
# dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
## Below batch_size calculation assumes the case without gradient accumulation.
## Manually set it to a lower value if you hit out of memory during training.
# batch_size=$(( ${global_batch_size} / ${dp_size} ))
###############################################################################
### Misc configs
log_interval=10
eval_iters=1
eval_interval=200
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=10
save_interval=$((${train_iters} / ${num_save}))

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="false"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

jobname="gebert-pile"
data_home="xxx"
TRAIN_DATA_PATH="xxx/train"
VALID_DATA_PATH="xxx/train"
TOKENIZER_PATH=xxx


jobname="${jobname}-${model_size}B-iters-${train_iters}"
jobname="${jobname}-lr-${lr}-min-${min_lr}-wmup-${lr_warmup_iters}-dcy-${lr_decay_iters}-sty-${lr_decay_style}"
jobname="${jobname}-gbs-${global_batch_size}-mbs-${batch_size}-gpu-${num_gpus}-zero-${zero_stage}-mp-${mp_size}-pp-${pp_size}"
if [ "${no_pp}" = "true" ]; then
    jobname="${jobname}-nopp"
fi

output_home="xxx"
log_path="${output_home}/log/"
checkpoint_path="${output_home}/${jobname}/checkpoint"
tensorboard_dir="${output_home}/project/bert_with_pile/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
###############################################################################
data_options=" \
    --train-data-path $TRAIN_DATA_PATH \
    --valid-data-path $VALID_DATA_PATH \
    --tokenizer-model $TOKENIZER_PATH \
    --tokenizer-type HFTokenizer \
    --data-impl mmap "

megatron_options=" \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std ${init_std} \
    --tensor-model-parallel-size ${mp_size} \
    --lr-decay-iters ${lr_decay_iters} \
    --lr-warmup-iters ${lr_warmup_iters} \
    --micro-batch-size ${batch_size} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --num-key-value-heads ${num_kv_nums} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${max_posi_emb} \
    --train-iters ${train_iters} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --save-interval ${save_interval} \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16 \
    --load ${checkpoint_path} \
    --save ${checkpoint_path}_new \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensorboard-dir ${tensorboard_path} \
    --swiglu \
    --use-rotary-position-embeddings \
    --rotary-percent 0.25 \
    --num-workers 16 \
    --distributed-backend nccl \
    --attention-softmax-in-fp32 "
    
###### if dpo training
#--dpo-trainingß
#--dpo-update-model-step
#-dpo-sampling-type
#--dpo-type


if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

config_json="../ds_config/ds_config_pretrain.json"

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --pipeline-model-parallel-size ${pp_size}"

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
    --deepspeed-activation-checkpointing"
fi


deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29000 ../../pretrain_gebert.py ${megatron_options} ${data_options} ${deepspeed_options}

#multi-node training
# torchrun $DISTRIBUTED_ARGS ../../pretrain_gebert.py ${megatron_options} ${data_options} ${deepspeed_options}


# >> ${log_path}/${jobname}_${current_time}.log
