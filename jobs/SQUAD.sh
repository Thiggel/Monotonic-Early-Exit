. environment.sh

# no early exit
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
    ../run_question_answering.py \
    --model_name_or_path ../checkpoints/SQUAD \
    --tokenizer_name t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5_large/ \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 5475 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --output_hidden_states_decoder True \
    --use_early_exit False

# softmax
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
    ../run_question_answering.py \
    --model_name_or_path ../checkpoints/SQUAD \
    --tokenizer_name t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5_large/ \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 5475 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --output_hidden_states_decoder True \
    --intermediate_loss_fn weighted_ce \
    --use_early_exit True \
    --exit_conf_type softmax \
    --exit_position_temp 4

# hidden_state_saturation
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
    ../run_question_answering.py \
    --model_name_or_path ../checkpoints/SQUAD \
    --tokenizer_name t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5_large/ \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 5475 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --output_hidden_states_decoder True \
    --intermediate_loss_fn weighted_ce \
    --use_early_exit True \
    --exit_conf_type hidden_state_saturation \
    --exit_position_temp 4

#last_three_prob_heuristic
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
    ../run_question_answering.py \
    --model_name_or_path ../checkpoints/SQUAD \
    --tokenizer_name t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5_large/ \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 5475 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --output_hidden_states_decoder True \
    --intermediate_loss_fn weighted_ce \
    --use_early_exit True \
    --exit_conf_type last_three_top_prob_heuristic \
    --exit_position_temp 4

# meta
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
    ../src/run_question_answering.py \
    --model_name_or_path ../src/model_checkpoints/t5-large \
    --tokenizer_name t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ../src/save/squad_t5_large/ \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 5475 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --output_hidden_states_decoder True \
    --use_early_exit True \
    --exit_conf_type meta \
    --exit_position_temp 4 \
    --exit_conf_threshold 0.5 \
    --do_train \
    --train_meta_cm_head \
    --num_train_epochs 5 \
    --deploy_scenario False 
    
# three hidden states classifier
!CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
    ../src/run_question_answering.py \
    --model_name_or_path ../src/model_checkpoints/t5-large \
    --tokenizer_name t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ../src/save/squad_t5_large/ \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 5475 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --output_hidden_states_decoder True \
    --use_early_exit True \
    --exit_position_temp 4 \
    --exit_conf_threshold 0.5 \
    --do_train \
    --train_meta_cm_head \
    --num_train_epochs 5 \
    --exit_conf_type last_three_hiddens_classifier \
    --deploy_scenario False 
    


# TODO: Set epochs!



# There are three of these commands missing in here:
# TODO: classifier - meta
# TODO: classifier - recurrent
# TODO: classifier - last_three_hidden_states
# After that, create an exact file like this for the two other datasets
# And after that, create three job files that only run these shell files
