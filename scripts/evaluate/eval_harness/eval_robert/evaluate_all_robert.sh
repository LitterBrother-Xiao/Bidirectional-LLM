
export HF_DATASETS_OFFLINE="1"
export CUDA_VISIBLE_DEVICES=3
python ../evaluate_all_roberta.py \
    --task_list arc_easy,arc_challenge,boolqnew,logiqa,sciq,winogrande,piqa,race,social_iqa \
    --inftype max_beam_based \
    --model_path ../roberta-large \
    --seq_length 510 