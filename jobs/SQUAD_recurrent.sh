. jobs/environment.sh

# recurrent classifier
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
    src/run_question_answering.py \
    --model_name_or_path checkpoints/SQUAD \
    --tokenizer_name t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir src/save/squad_t5_large/ \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 5475 \
    --learning_rate 1e-4 \
    --output_hidden_states_decoder True \
    --use_early_exit True \
    --exit_conf_type recurrent_classifier \
    --exit_position_temp 4 \
    --exit_conf_threshold 0.7 \
    --do_train \
    --train_meta_cm_head \
    --num_train_epochs 5 \
    --deploy_scenario False \
    --dataloader_drop_last True \
    --max_train_samples 320 \
    --max_eval_samples 160
