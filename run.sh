
asr_config=checkpoints/CTC/conformerL_ctc_baseline2048/config.yaml
model_path=checkpoints/CTC/conformerL_ctc_baseline2048/latest.pth
audio_path=record/house

python3 main.py \
    --asr_config $asr_config \
    --model_path $model_path \
    --audio_path $audio_path