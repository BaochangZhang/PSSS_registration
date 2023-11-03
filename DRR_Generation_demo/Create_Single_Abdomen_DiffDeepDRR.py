from libs.DiffDeepDRR.Differentiable_DRRs import Differentiable_DRRs
from libs.DiffDeepDRR.vol.volume_Realistic import Volume_Realistc
from libs.DiffDeepDRR.drr_projectors.proj_zbc import Deepdrrbased_Projector
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import time
from PIL import Image
import cv2
from glob import glob


def test_PatientCT_DRR():

    vol = Volume_Realistc.from_nifti(filepath=str('../testdata/ct.nii'),
                                     resample=True, resample_spacing=[2.0, 2.0, 2.0],
                                     HU_segments=[-800, 350], target_orient='RIA',
                                     spectrum='90KV_AL40', use_cache=False)
    vol.Update()
    assert vol.check_ready(), f"please call vol.Update()"
   
    # Make the DRR Engine
    Proj = Deepdrrbased_Projector(vol, step=max(vol.get_spacing()))
    drr = Differentiable_DRRs(Vol=vol, Projector=Proj, Target_id=None,
                              detector_center_x=216.0, detector_center_y=216.0,
                              height=256, pixel_size=1.6875,
                              normlized=False, bone_dark=True)

    alpha, beta, gamma, tx, ty, tz = [0, 0, 0, 0, 0, 0]
    print('pose:', alpha, beta, gamma, tx, ty, tz)

    img, _ = drr(alpha, beta, gamma, tx, ty, tz)
    DRR_img = np.squeeze(img[0, :, :].detach().cpu().numpy())
    plt.imshow(DRR_img, cmap='gray', vmax=1, vmin=0)
    plt.show()


if __name__ == '__main__':
    test_PatientCT_DRR()
    