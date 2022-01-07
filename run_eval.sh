CUDA_VISIBLE_DEVICES=2 python eval.py \
    --model_name_or_path=save/jonatasgrosman/wav2vec2-large-xlsr-53-english/checkpoint-22212 \
    --train_manifest_path=dataset/train_metadata.csv \
    --valid_manifest_path=dataset/validation_metadata.csv \
    --test_manifest_path=dataset/test_metadata.csv \
    --cache_dir_name cache/jonatasgrosman/wav2vec2-large-xlsr-53-english \
    --preprocessing_num_workers=16 \
    --audio_column_name=file_name --text_column_name=transcription \
    --eval_accumulation_steps=1 \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
    --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
    --seed=14045 --num_train_epochs=5 --learning_rate=1e-5 --output_dir=./eval/en
    
CUDA_VISIBLE_DEVICES=3 python eval.py \
    --model_name_or_path=save/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn/checkpoint-19127 \
    --train_manifest_path=dataset/train_metadata.csv \
    --valid_manifest_path=dataset/validation_metadata.csv \
    --test_manifest_path=dataset/test_metadata.csv \
    --cache_dir_name cache/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
    --preprocessing_num_workers=16 \
    --audio_column_name=file_name --text_column_name=transcription \
    --eval_accumulation_steps=1 \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=16 \
    --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
    --seed=14045 --num_train_epochs=5 --learning_rate=1e-5 --output_dir=./eval/zh