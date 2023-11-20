# export PYTHONPATH=./
# export CUDA_VISIBLE_DEVICES=0

$env:PYTHONPATH = ".\"
# $env:CUDA_VISIBLE_DEVICES = "0"
$Video_ID = "gdg8"  # 替换 YourVideoID 为你的视频ID

# 1. extrac 16khz wav
python data_util/process.py --video_id=$Video_ID --task=1
# 2. extrac deepspeech and esperanto; 3.extract image frames
Start-Job -ScriptBlock { python data_util/process.py --video_id=$using:Video_ID --task=2 }
Start-Job -ScriptBlock { python data_util/process.py --video_id=$using:Video_ID --task=3 }
# 等待任务完成
Get-Job | Wait-Job | Receive-Job

# 7.detect landmarks
python data_util/process.py --video_id=$Video_ID --task=7
# 4.face segmentation parsing; 8.estimate head pose
Start-Job -ScriptBlock { python data_util/process.py --video_id=$using:Video_ID --task=4 }
Start-Job -ScriptBlock { python data_util/process.py --video_id=$using:Video_ID --task=8 }
# 等待任务完成
Get-Job | Wait-Job | Receive-Job

# 4. extract background image
python data_util/process.py --video_id=$Video_ID --task=5
# Optional: Once the background image is extracted before running step 5,
# you could use an image inpainting tool (such as Inpaint on MacOS)
# to edit the background image, so it could be more realistic.
# 5. save head, torso, gt imgs
python data_util/process.py --video_id=$Video_ID --task=6

# 7. integrate the results into meta
python data_util/process.py --video_id=$Video_ID --task=9
# 8. calculate audio features
python data_gen/nerf/extract_hubert_mel_f0.py --video_id=$Video_ID
# 9. calculate 3DMM
python data_gen/nerf/extract_3dmm.py --video_id=$Video_ID
# binarize the dataset into `data/binary/videos/$Video_ID/trainval_dataset.npy`
python data_gen/nerf/binarizer.py --config=egs/datasets/videos/$Video_ID/lm3d_radnerf.yaml

# 1. extrac 16khz wav
python data_util/process.py --video_id=gdg8 --task=1
# 2. extrac deepspeech and esperanto; 3.extract image frames 
python data_util/process.py --video_id=gdg8 --task=2 
python data_util/process.py --video_id=gdg8 --task=3
# 7.detect landmarks
python data_util/process.py --video_id=gdg8 --task=7
# 4.face segmentation parsing; 8.estimate head pose
python data_util/process.py --video_id=gdg8 --task=4 
python data_util/process.py --video_id=gdg8 --task=8
# 4. extract background image
python data_util/process.py --video_id=gdg8 --task=5
# Optional: Once the background image is extracted before running step 5,
# you could use a image inpainting tool (such as Inpaint on MacOS)
# to edit the backgroud image, so it could be more realistic.
# 5. save head, torso, gt imgs
python data_util/process.py --video_id=gdg8 --task=6
wait
# 7. integrate the results into meta
python data_util/process.py --video_id=gdg8 --task=9
# 8. calculate audio features
python data_gen/nerf/extract_hubert_mel_f0.py --video_id=gdg8
# 9. calculate 3DMM 
python data_gen/nerf/extract_3dmm.py --video_id=gdg8
# binarize the dataset into `data/binary/videos/gdg8/trainval_dataset.npy`
python data_gen/nerf/binarizer.py --config=egs/datasets/videos/gdg8/lm3d_radnerf.yaml

python tasks/run.py --config=egs/datasets/videos/gdg8/lm3d_postnet_sync.yaml --exp_name=gdg8/postnet
python tasks/run.py --config=egs/datasets/videos/gdg8/lm3d_radnerf.yaml --exp_name=gdg8/lm3d_radnerf
python tasks/run.py --config=egs/datasets/videos/gdg8/lm3d_radnerf_torso.yaml --exp_name=gdg8/lm3d_radnerf_torso


python data_util/process.py --video_id=gdg8 --task=7
# 4.face segmentation parsing; 8.estimate head pose
python data_util/process.py --video_id=gdg8 --task=4 
python -c "print('success!')"
