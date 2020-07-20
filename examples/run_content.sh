
#######################################################################  rows + max
CUDA_VISIBLE_DEVICES=0,1 python run_table.py \
 --data_dir '../data/' \
 --model_name_or_path '/home/colozoy/data/pre_trained/bert-large-cased'\
 --fasttext_dir '/home/colozoy/data/pre_trained/fasttext/wiki.simple.bin'\
 --model_type 'bert' \
 --use_caption  \
 --use_pg_title \
 --use_sec_title \
 --use_schema \
 --content 'ROW' \
 --selector 'MAX' \
 --output_dir '/home/colozoy/data/IR/table/table/outputs/bert-large-cased' \
 --task_name 'table' \
 --max_seq_length 128 \
 --k_fold 5 \
 --overwrite_cache \
 --overwrite_output_dir \
 --fold 1 \
 --schema 'SEP' \
 --do_train \
 --do_eval \
 --evaluate_during_training \
 --per_gpu_train_batch_size 8 \
 --per_gpu_eval_batch_size 8 \
 --gradient_accumulation_steps 1 \
 --learning_rate 1e-5 \
 --num_train_epochs 5 \
 --logging_steps 10 \
 --save_steps 10 \
 --warmup_proportion 0.1 \
 --seed 777 \





