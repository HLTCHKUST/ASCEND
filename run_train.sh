# Wav2Vec 2.0 English
CUDA_VISIBLE_DEVICES=0 python train.py --model_name_or_path=jonatasgrosman/wav2vec2-large-xlsr-53-english \
    --train_manifest_path=dataset/train_metadata.csv \
    --valid_manifest_path=dataset/validation_metadata.csv \
    --test_manifest_path=dataset/test_metadata.csv \
    --preprocessing_num_workers=16 --audio_column_name=file_name --text_column_name=transcription \
    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
    --dataloader_num_workers=16 --dataloader_pin_memory --group_by_length \
    --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
    --fp16 --fp16_backend=amp \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
    --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
    --metric_for_best_model=mer --greater_is_better=False \
    --gradient_checkpointing=True
    
# Wav2Vec 2.0 Chinese
CUDA_VISIBLE_DEVICES=0 python train.py --model_name_or_path=jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
    --train_manifest_path=dataset/train_metadata.csv \
    --valid_manifest_path=dataset/validation_metadata.csv \
    --test_manifest_path=dataset/test_metadata.csv \
    --preprocessing_num_workers=16 --audio_column_name=file_name --text_column_name=transcription \
    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
    --dataloader_num_workers=16 --dataloader_pin_memory --group_by_length \
    --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
    --fp16 --fp16_backend=amp \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
    --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
    --metric_for_best_model=mer --greater_is_better=False \
    --gradient_checkpointing=True