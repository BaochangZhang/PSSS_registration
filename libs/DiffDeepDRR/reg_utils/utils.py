import torch
import numpy as np


def parse_optimizer(optimizer, drr_moving, T_lr=None, R_lr=None):
    """Get the optimizer."""
    if optimizer == "adam":
        return torch.optim.Adam(
            [
                {"params": [drr_moving.rotations], "lr": T_lr},
                {"params": [drr_moving.translations], "lr": R_lr},
            ],
        )
    elif optimizer == "lbfgs":
        return torch.optim.LBFGS(
            [drr_moving.rotations, drr_moving.translations],
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def run_convergence_adam_noplt(
        fixed_img,
        drr_moving,
        cost_function,
        T_lr,
        R_lr,
        n_itrs,
        early_stop_v=1e-4):

    optimizer = parse_optimizer("adam", drr_moving, T_lr, R_lr)
    lr_scheduler_strategy = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.50)

    best_score = 1e9
    best_est_pose = []
    # Run the optimization loop
    for itr in range(1, n_itrs + 1):
        # Forward pass: compute the moving DRR
        estimate, _ = drr_moving()
        # Compute the loss
        loss = cost_function(estimate, fixed_img)
        # print('itr:%d, loss:%f' % (itr, loss.item()))
        if best_score > loss:
            # print('itr:%d, loss:%f' % (itr, loss.item()))
            best_score = loss.item()
            est_pose_t = np.squeeze(drr_moving.translations.data.detach().cpu().numpy())
            est_pose_r = np.squeeze(drr_moving.rotations.data.detach().cpu().numpy())
            tx, ty, tz = est_pose_t.tolist()
            alpha, beta, gama = est_pose_r.tolist()
            best_est_pose = [alpha, beta, gama, tx, ty, tz]

            if best_score <= early_stop_v:
                break
        # Backward pass: update the 6DoF parameters
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        lr_scheduler_strategy.step()

    return best_est_pose, best_score


def run_convergence_lbfgs_noplt(
        fixed_img,
        drr_moving,
        cost_function,
        n_itrs):

    # Get the loss function and optimizer
    optimizer = parse_optimizer('lbfgs', drr_moving)

    best_score = 1e9
    best_est_pose = []

    # Run the optimization loop
    for itr in range(1, n_itrs + 1):

        def loss_closure():
            optimizer.zero_grad()
            estimate, _ = drr_moving()
            loss = cost_function(estimate, fixed_img)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(loss_closure)
        loss = loss_closure()
        if best_score > loss:
            # print('itr:%d, loss:%f' % (itr, loss.item()))
            best_score = loss.item()
            est_pose_t = np.squeeze(drr_moving.translations.data.detach().cpu().numpy())
            est_pose_r = np.squeeze(drr_moving.rotations.data.detach().cpu().numpy())
            tx, ty, tz = est_pose_t.tolist()
            alpha, beta, gama = est_pose_r.tolist()
            best_est_pose = [alpha, beta, gama, tx, ty, tz]

    return best_est_pose, best_score
