export CUDA_VISIBLE_DEVICES=0
export Video_ID=gdg6_2
export Wav_ID=000

$env:Video_ID = "gdg6_2"
$env:Wav_ID = "000"

python inference/audio2pose/audio2pose_infer.py \
    --config=checkpoints/${Video_ID}/audio2pose/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
infer_out_npy_name=infer_out/${Video_ID}/pred_c2w/${Wav_ID}.npy \
    --reset

python.exe inference\audio2pose\audio2pose_infer.py `
--config="checkpoints\$env:Video_ID\audio2pose\config.yaml" `
--hparams="infer_audio_source_name=data\raw\val_wavs\$env:Wav_ID.wav, infer_out_npy_name=infer_out\$env:Video_ID\pred_c2w\$env:Wav_ID.npy" `
--reset