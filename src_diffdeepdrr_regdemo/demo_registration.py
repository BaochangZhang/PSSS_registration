import numpy as np
from libs.DiffDeepDRR.Differentiable_DRRs import Differentiable_DRRs
from libs.DiffDeepDRR.vol.volume_Realistic import Volume_Realistc
from libs.DiffDeepDRR.drr_projectors.proj_zbc import Deepdrrbased_Projector
from libs.DiffDeepDRR.reg_utils.utils import run_convergence_lbfgs_noplt, run_convergence_adam_noplt
from libs.network_utils.loss_func import ZNCC
import matplotlib.pyplot as plt


def DRR_config(vol_dir, device, HU_segments=[-800, 350]):
    # Make the DRR Engine
    vol = Volume_Realistc.from_nifti(filepath=vol_dir,
                                     resample=True, resample_spacing=[2.0, 2.0, 2.0],
                                     HU_segments=HU_segments, target_orient='RIA',
                                     spectrum='90KV_AL40', use_cache=False)
    vol.Update()
    assert vol.check_ready(), f"please call vol.Update()"
    # Make the DRR Engine
    Proj = Deepdrrbased_Projector(vol, step=max(vol.get_spacing()), device=device)
    drr = Differentiable_DRRs(Vol=vol, Projector=Proj, device=device,
                              detector_center_x=216.0, detector_center_y=216.0,
                              height=256, pixel_size=1.6875, out_dim=4,
                              normlized=True, bone_dark=True)
    return drr


def get_true_drr(max_T=30, max_R=15):
    """Get parameters for the fixed DRR."""
    tx = np.around(np.random.uniform(-max_T, max_T), decimals=1)
    ty = np.around(np.random.uniform(-max_T, max_T), decimals=1)
    tz = np.around(np.random.uniform(-max_T, max_T), decimals=1)
    alpha = np.around(np.random.uniform(-max_R, max_R), decimals=1)
    beta = np.around(np.random.uniform(-max_R, max_R), decimals=1)
    gamma = np.around(np.random.uniform(-10.0, 10.0), decimals=1)
    gt_pose = [alpha, beta, gamma, tx, ty, tz]
    return gt_pose


def get_image_gradient(arr, scale=1.0) -> object:
    gx1, gy1 = np.gradient(arr)
    edge = (gx1 ** 2 + gy1 ** 2) ** 0.5
    edge = edge / (np.percentile(edge, 99))
    edge = np.tanh(edge * scale)
    return edge


def reg_demo():
    drr_moving = DRR_config('../testdata/ct.nii',
                            device='cuda:0')
    fixed_pose = get_true_drr()
    alpha, beta, gamma, bx, by, bz = fixed_pose
    fixed_img, _ = drr_moving(alpha, beta, gamma, bx, by, bz)

    init_pose = list(np.asarray(fixed_pose) - 5.0)
    alpha, beta, gamma, bx, by, bz = init_pose
    init_img, _ = drr_moving(alpha, beta, gamma, bx, by, bz)

    cost_func = ZNCC().to('cuda:0')
    # Run the optimization
    refine_pose, _ = run_convergence_adam_noplt(fixed_img, drr_moving, cost_func, T_lr=5.0, R_lr=0.5, n_itrs=100)
    # refine_pose, _ = run_convergence_lbfgs_noplt(fixed_img, drr_moving, cost_func, n_itrs=5)

    print('-----------------------------------------')
    print('groundtruth_pose:', fixed_pose)
    init_error = np.asarray(fixed_pose) - np.asarray(init_pose)
    print('init_pred_pose:', init_pose, 'R_error:', np.linalg.norm(init_error[:3]), 'T_error:', np.linalg.norm(init_error[3:]))
    ref_error = np.asarray(fixed_pose) - np.asarray(refine_pose)
    print('final_pred_pose:', refine_pose, 'R_error:', np.linalg.norm(ref_error[:3]), 'T_error:', np.linalg.norm(ref_error[3:]))
    print('-----------------------------------------')

    alpha, beta, gamma, bx, by, bz = list(refine_pose)
    final_img, _ = drr_moving(alpha, beta, gamma, bx, by, bz)

    final_img = np.squeeze(final_img[0, :, :].detach().cpu().numpy())
    fixed_img = np.squeeze(fixed_img[0, :, :].detach().cpu().numpy())
    init_img = np.squeeze(init_img[0, :, :].detach().cpu().numpy())

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    axs[0, 0].imshow(fixed_img, cmap='gray')
    axs[0, 0].set_title('fixed-img')
    # axs[0, 0].axis('off')

    axs[0, 1].imshow(final_img, cmap='gray')
    axs[0, 1].set_title('reg-img')
    # axs[0, 1].axis('off')

    edg_gt = get_image_gradient(fixed_img)
    edg_gt = (edg_gt-edg_gt.min())/(edg_gt.max()-edg_gt.min())
    edg_gt = edg_gt-edg_gt.mean()
    edg_gt[edg_gt<0]=0

    edg_final = get_image_gradient(final_img)
    edg_final = (edg_final-edg_final.min())/(edg_final.max()-edg_final.min())
    edg_final = edg_final-edg_final.mean()
    edg_final[edg_final<0]=0

    edg_init = get_image_gradient(init_img)
    edg_init = (edg_init-edg_init.min())/(edg_init.max()-edg_init.min())
    edg_init = edg_init-edg_init.mean()
    edg_init[edg_init<0]=0

    axs[1, 0].imshow(edg_gt, cmap='Reds')
    axs[1, 0].imshow(edg_init, alpha=0.6, cmap='Blues')
    axs[1, 0].set_title('init.')
    # axs[1, 0].axis('off')
    axs[1, 1].imshow(edg_gt, cmap='Reds')
    axs[1, 1].imshow(edg_final, alpha=0.6, cmap='Blues')
    axs[1, 1].set_title('final')
    # axs[1, 1].axis('off')

    plt.show()


if __name__ == "__main__":
    reg_demo()

