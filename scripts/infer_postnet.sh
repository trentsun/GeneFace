export CUDA_VISIBLE_DEVICES=0
export Video_ID=gdgt6_2
export Wav_ID=000
export Postnet_Ckpt_Steps=4000 # please reach to `docs/train_models.md` to get some tips about how to select an approprate ckpt_steps!

python inference/postnet/postnet_infer.py \
    --config=checkpoints/${Video_ID}/lm3d_postnet_sync/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
infer_out_npy_name=infer_out/${Video_ID}/pred_lm3d/${Wav_ID}.npy,\
infer_ckpt_steps=${Postnet_Ckpt_Steps} \
    --reset

$env:CUDA_VISIBLE_DEVICES = "0"
$env:Video_ID = "gdg6_2"
$env:Wav_ID = "guodegang_wav_20231114214421"
$env:Postnet_Ckpt_Steps = "6000"
$env:PYTHONPATH = ".\"



python.exe inference\\postnet\\postnet_infer.py `
    --config=checkpoints\\$env:Video_ID\\postnet\\config.yaml `
    --hparams="infer_audio_source_name=data\raw\val_wavs\\$env:Wav_ID.wav,infer_out_npy_name=infer_out\\$env:Video_ID\pred_lm3d\\$env:Wav_ID.npy, infer_ckpt_steps=$env:Postnet_Ckpt_Steps" `
    --reset

python .\inference\postnet\postnet_infer.py --config=.\checkpoints\gdg6_2\postnet\config.yaml --hparams="infer_audio_source_name=data\raw\val_wavs\$env:Wav_ID.wav,infer_out_npy_name=infer_out\$env:Video_ID\pred_lm3d\$env:Wav_ID.npy,infer_ckpt_steps=$env:Postnet_Ckpt_Steps" --reset



$paramString = "--config=checkpoints\\$env:Video_ID\\postnet\\config.yaml " +
               "--hparams='infer_audio_source_name=data/raw/val_wavs/$env:Wav_ID.wav, infer_out_npy_name=infer_out/$env:Video_ID/pred_lm3d/$env:Wav_ID.npy, infer_ckpt_steps=$env:Postnet_Ckpt_Steps' " +
               "--reset"

python.exe inference\postnet\postnet_infer.py $paramString