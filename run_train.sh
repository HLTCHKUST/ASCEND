CUDA_VISIBLE_DEVICES=3 python train.py --model_name_or_path=ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt \
    --train_manifest_path=dataset/train_metadata.csv \
    --valid_manifest_path=dataset/validation_metadata.csv \
    --test_manifest_path=dataset/test_metadata.csv \
    --num_workers=8 --preprocessing_num_workers=16 \
    --audio_column_name=file_name --text_column_name=transcription \
    --per_device_train_batch_size=4 --per_device_eval_batch_size=8 \
    --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
    --seed=14045 --num_train_epochs=5 --learning_rate=1e-5 \
    --fp16 --fp16_backend=amp \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=100 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=3 \
    --metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True \
    --gradient_checkpointing=True \

# CUDA_VISIBLE_DEVICES=2 python train.py --model_name_or_path=jonatasgrosman/wav2vec2-large-xlsr-53-english \
#     --train_manifest_path=dataset/train_metadata.csv \
#     --valid_manifest_path=dataset/validation_metadata.csv \
#     --test_manifest_path=dataset/test_metadata.csv \
#     --num_workers=8 --preprocessing_num_workers=16 \
#     --audio_column_name=file_name --text_column_name=transcription \
#     --per_device_train_batch_size=4 --per_device_eval_batch_size=8 \
#     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
#     --seed=14045 --num_train_epochs=5 --learning_rate=1e-5 \
#     --fp16 --fp16_backend=amp \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=100 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=3 \
#     --metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True \
#     --gradient_checkpointing=True \

# # Distributed
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train.py --model_name_or_path=ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt     --train_manifest_path=dataset/train_metadata.csv     --valid_manifest_path=dataset/validation_metadata.csv     --test_manifest_path=dataset/test_metadata.csv     --num_workers=8 --preprocessing_num_workers=8     --audio_column_name=file_name --text_column_name=transcription     --per_device_train_batch_size=4 --per_device_eval_batch_size=4     --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length  --seed=14045 --num_train_epochs=20 --learning_rate=1e-5 --fp16 --fp16_backend=amp --sharded_ddp=simple
