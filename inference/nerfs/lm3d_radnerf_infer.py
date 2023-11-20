import torch
import numpy as np

from utils.commons.hparams import hparams

from tasks.radnerfs.dataset_utils import RADNeRFDataset
from inference.nerfs.lm3d_nerf_infer import LM3dNeRFInfer
from data_util.face3d_helper import Face3DHelper


class LM3d_RADNeRFInfer(LM3dNeRFInfer):
    def __init__(self, hparams, device=None):
        super().__init__(hparams, device)
        self.dataset_cls = RADNeRFDataset # the dataset only provides head pose 
        self.dataset = self.dataset_cls('trainval', training=False)
        self.face3d_helper = Face3DHelper()

    def get_pose_from_ds(self, samples):
        """
        process the item into torch.tensor batch
        """
        for i, sample in enumerate(samples):
            ds_sample = self.dataset[i]
            sample['rays_o'] = ds_sample['rays_o']
            sample['rays_d'] = ds_sample['rays_d']
            sample['bg_coords'] = ds_sample['bg_coords']
            sample['pose'] = ds_sample['pose']
            sample['idx'] = ds_sample['idx']
            sample['bg_img'] = ds_sample['bg_img']
            sample['H'] = ds_sample['H']
            sample['W'] = ds_sample['W']
        return samples

    def get_cond_from_input(self, inp):
        """
        :param inp: {'audio_source_name': (str), 'cond_name': (str, optional)}
        :return: a list that contains the condition feature of NeRF
        """
        self.save_wav16k(inp)

        # load the lm3d as the condition for lm3d head nerf
        assert inp['cond_name'].endswith('.npy')
        lm3d_arr = np.load(inp['cond_name'])[0] # [T, w=16, c=29]
        idexp_lm3d = torch.from_numpy(lm3d_arr).float()
        print(f"Loaded pre-extracted 3D landmark sequence from {inp['cond_name']}!")
        # idexp_lm3d = self.face3d_helper.close_eyes_for_idexp_lm3d(idexp_lm3d)
        # idexp_lm3d = self.face3d_helper.close_mouth_for_idexp_lm3d(idexp_lm3d)

        idexp_lm3d_mean = self.dataset.idexp_lm3d_mean
        idexp_lm3d_std = self.dataset.idexp_lm3d_std
        idexp_lm3d_normalized = (idexp_lm3d.reshape([-1,68,3]) - idexp_lm3d_mean)/idexp_lm3d_std

        # step3. clamp the lm3d, to regularize apparent outliers
        lm3d_clamp_std = hparams['infer_lm3d_clamp_std']
        idexp_lm3d_normalized[:,0:17] = torch.clamp(idexp_lm3d_normalized[:,0:17], -lm3d_clamp_std, lm3d_clamp_std) # yaw_x_y_z
        idexp_lm3d_normalized[:,17:27,0:2] = torch.clamp(idexp_lm3d_normalized[:,17:27,0:2], -lm3d_clamp_std/2, lm3d_clamp_std/2) # brow_x_y
        idexp_lm3d_normalized[:,17:27,2] = torch.clamp(idexp_lm3d_normalized[:,17:27,2], -lm3d_clamp_std, lm3d_clamp_std) # brow_z
        idexp_lm3d_normalized[:,27:36] = torch.clamp(idexp_lm3d_normalized[:,27:36], -lm3d_clamp_std, lm3d_clamp_std) # nose
        idexp_lm3d_normalized[:,36:48,0:2] = torch.clamp(idexp_lm3d_normalized[:,36:48,0:2], -lm3d_clamp_std/2, lm3d_clamp_std/2) # eye_x_y
        idexp_lm3d_normalized[:,36:48,2] = torch.clamp(idexp_lm3d_normalized[:,36:48,2], -lm3d_clamp_std, lm3d_clamp_std) # eye_z
        idexp_lm3d_normalized[:,48:68] = torch.clamp(idexp_lm3d_normalized[:,48:68], -lm3d_clamp_std, lm3d_clamp_std) # mouth
        
        _lambda_other = 0.2
        _lambda_lip = 0.2
        moving_lm = idexp_lm3d_normalized[0].clone()
        for i in range(len(idexp_lm3d_normalized)):
            idexp_lm3d_normalized[i,0:17] = _lambda_other * moving_lm[0:17] + (1 - _lambda_other) * idexp_lm3d_normalized[i,0:17] # yaw
            idexp_lm3d_normalized[i,17:27] = _lambda_other * moving_lm[17:27] + (1 - _lambda_other) * idexp_lm3d_normalized[i,17:27] # brow
            idexp_lm3d_normalized[i,27:36] = _lambda_other * moving_lm[27:36] + (1 - _lambda_other) * idexp_lm3d_normalized[i,27:36] # nose
            idexp_lm3d_normalized[i,36:48] = _lambda_other * moving_lm[36:48] + (1 - _lambda_other) * idexp_lm3d_normalized[i,36:48] # eye
            idexp_lm3d_normalized[i,48:68] = _lambda_lip * moving_lm[48:68] + (1 - _lambda_lip) * idexp_lm3d_normalized[i,48:68]
            moving_lm.data = idexp_lm3d_normalized[i].data

        idexp_lm3d_normalized = idexp_lm3d_normalized.reshape([-1,68*3])
        from data_gen.nerf.binarizer import get_win_conds
        idexp_lm3d_normalized_numpy = idexp_lm3d_normalized.cpu().numpy()
        idexp_lm3d_normalized_win_numpy = np.stack([get_win_conds(idexp_lm3d_normalized_numpy, i, smo_win_size=hparams['cond_win_size'], pad_option='edge') for i in range(idexp_lm3d_normalized_numpy.shape[0])])
        idexp_lm3d_normalized_win = torch.from_numpy(idexp_lm3d_normalized_win_numpy)

        samples = [{} for _ in range(len(idexp_lm3d_normalized))]
        for idx, sample in enumerate(samples):
            sample['cond'] = idexp_lm3d_normalized[idx].unsqueeze(0)
            if hparams['use_window_cond']:
                sample['cond_win'] = idexp_lm3d_normalized_win[idx]
                sample['cond_wins'] = torch.from_numpy(get_win_conds(idexp_lm3d_normalized_win_numpy, idx, hparams['smo_win_size'], 'edge'))
        return samples

def infer_main(config='E:\\aigc\\GeneFace\\checkpoints\\gdg6_2\\lm3d_radnerf_torso\\config.yaml', hparams_str='infer_audio_source_name=data\\raw\\val_wavs\\000.wav,infer_cond_name=infer_out/gdg6_2/pred_lm3d/000.npy,infer_out_video_name=infer_out/gdg6_2/pred_video/000_radnerf_torso_smo.mp4,exp_name=gdg6_2/lm3d_nerf_torso'):
    import os
    os.environ['config'] = config
    os.environ['exp_name'] = 'gdg6_2/lm3d_radnerf_torso'
    os.environ['hparams_str'] = hparams_str
    from utils.commons.hparams import hparams, set_hparams

    # from utils.commons.hparams import hparams as hp
    # set_hparams(config=config, hparams_str=hparams_str, exp_name='gdg6_2/lm3d_nerf_torso')
    set_hparams
    inp = {
            'audio_source_name': 'data/raw/val_wavs/000_16k.wav',
            'cond_name': 'infer_out/gdg6_2/pred_lm3d/000.npy',
            'out_video_name': 'infer_out/gdg6_2/pred_video/000.mp4',
            }

    LM3d_RADNeRFInfer.example_run(inp)

if __name__ == '__main__':
    from utils.commons.hparams import set_hparams
    from utils.commons.hparams import hparams as hp
    inp = {
            'audio_source_name': 'data/raw/val_wavs/000_16k.wav',
            'cond_name': 'infer_out/gdg6_2/pred_lm3d/000.npy',
            'out_video_name': 'infer_out/gdg6_2/pred_video/000.mp4',
            }

    LM3d_RADNeRFInfer.example_run(inp)