export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1 # now we support multi-gpu inference!
export Video_ID=gdg6_2
export Wav_ID=000 # the .wav file should locate at `data/raw/val_wavs/<wav_id>.wav`

python inference/nerfs/lm3d_radnerf_infer.py \
    --config=checkpoints/${Video_ID}/lm3d_radnerf_torso/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
infer_cond_name=infer_out/${Video_ID}/pred_lm3d/${Wav_ID}.npy,\
infer_out_video_name=infer_out/${Video_ID}/pred_video/${Wav_ID}_radnerf_torso_smo.mp4\
    --infer

# python inference/nerfs/lm3d_radnerf_infer.py `
#     --config=checkpoints/${Video_ID}/lm3d_radnerf_torso/config.yaml `
#     --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,`
# infer_cond_name=infer_out/${Video_ID}/pred_lm3d/${Wav_ID}.npy,`
# infer_out_video_name=infer_out/${Video_ID}/pred_video/${Wav_ID}_radnerf_torso_smo.mp4 `
#     --infer

infer_audio_source_name=data\\raw\\val_wavs\\{wav_id}.wav,infer_cond_name=infer_out/gdg6_2/pred_lm3d/{wav_id}.npy,infer_out_video_name=infer_out/gdg6_2/pred_video/{wav_id}_radnerf_torso_smo.mp

hparams='infer_audio_source_name=data\\raw\\val_wavs\\000.wav,infer_cond_name=infer_out/gdg6_2/pred_lm3d/000.npy,infer_out_video_name=infer_out/gdg6_2/pred_video/000_radnerf_torso_smo.mp4'

python inference/nerfs/lm3d_radnerf_infer.py `
    --config=checkpoints/\$env:Video_ID/lm3d_radnerf_torso/config.yaml `
    --hparams=infer_audio_source_name=data/raw/val_wavs/\$env:Wav_ID.wav,`
infer_cond_name=infer_out/\$env:Video_ID/pred_lm3d/\$env:Wav_ID.npy,`
infer_out_video_name=infer_out/\$env:Video_ID/pred_video/\$env:Wav_ID_radnerf_torso_smo.mp4 `
    --infer
