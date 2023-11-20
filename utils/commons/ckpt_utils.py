import glob
import os
import re
import torch


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    print(ckpt_paths)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern),
                  key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))


def load_ckpt(cur_model, ckpt_base_dir, model_name='model', force=True, strict=True, steps=None):
    print("ckpt_base_dir", flush=True)
    print(f'{cur_model}, {ckpt_base_dir}, {model_name}, {force}, {strict}, {steps}', flush=True)
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        ckpt_path = ckpt_base_dir
        checkpoint = torch.load(ckpt_base_dir, map_location='cpu')
    else:
        base_dir = ckpt_base_dir
        checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)
    if checkpoint is not None:
        state_dict = checkpoint["state_dict"]
        if len([k for k in state_dict.keys() if '.' in k]) > 0:
            state_dict = {k[len(model_name) + 1:]: v for k, v in state_dict.items()
                          if k.startswith(f'{model_name}.')}
        else:
            if '.' not in model_name:
                state_dict = state_dict[model_name]
            else:
                base_model_name = model_name.split('.')[0]
                rest_model_name = model_name[len(base_model_name) + 1:]
                state_dict = {
                    k[len(rest_model_name) + 1:]: v for k, v in state_dict[base_model_name].items()
                    if k.startswith(f'{rest_model_name}.')}
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys: ", key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        cur_model.load_state_dict(state_dict, strict=strict)
        print(f"| load '{model_name}' from '{ckpt_path}'.")
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)

# os.chdir('.\GeneFace')
# os.environ['PYTHONPATH'] = '.\GeneFace'
# print(os.getcwd())
# a, b = get_last_checkpoint('E:/aigc/Geneface/checkpoints/gdg6_2/lm3d_radnerf_torso', 250000)
# print(a)
# print(b)

# import os

# # Directory to search in
# directory = "E:/aigc/Geneface/checkpoints/gdg6_2/lm3d_radnerf_torso"

# # File pattern to match
# file_pattern = "model_ckpt_steps_250000.ckpt"

# # Find all files that match the pattern
# matching_files = []
# for root, dirs, files in os.walk(directory):
#     for file in files:
#         if file_pattern in file:
#             matching_files.append(os.path.join(root, file))
# print(matching_files)
# # Sort the files to find the one with the highest number
# if matching_files:
#     latest_file = max(matching_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
# else:
#     latest_file = None

# print(latest_file)