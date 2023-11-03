import torch
import copy

class Deepdrrbased_Projector:
    def __init__(self, V, dtype=torch.float32, device="cuda", step=1.0, eps=1e-16):

        self.dtype = dtype
        self.device = device
        self.eps = eps
        vol = torch.tensor(V.get_data(), dtype=self.dtype, device=self.device)
        mats = torch.tensor(V.get_materials(), dtype=self.dtype, device=self.device)

        self.num_materials = mats.shape[0]
        self.material_grid = mats.permute(0, 3, 2, 1)[None, :]
        self.volume_grid = vol.permute(2, 1, 0)[None, None, :]

        self.energy = torch.tensor(V.get_spectrum_kev(), dtype=self.dtype, device=self.device)
        self.energy_pdf = torch.tensor(V.get_spectrum_pdf(), dtype=self.dtype, device=self.device)
        self.absorption_coef = torch.tensor(V.get_absorption_coef(), dtype=self.dtype, device=self.device)

        self.spacing = torch.tensor(V.get_spacing(), dtype=self.dtype, device=self.device)
        self.dims = torch.tensor(vol.shape, dtype=self.dtype, device=self.device)
        self.step = torch.tensor(step, dtype=self.dtype, device=self.device)

    def get_alpha_minmax(self, source, target):
        sdd = target - source + self.eps
        planes = torch.zeros(3, device=self.device) - 0.5
        alpha0 = (planes * self.spacing - source) / sdd
        planes = self.dims - 0.5
        alpha1 = (planes * self.spacing - source) / sdd
        alphas = torch.stack([alpha0, alpha1])

        alphamin = alphas.min(dim=0).values.max(dim=-1).values.unsqueeze(-1)
        alphamax = alphas.max(dim=0).values.min(dim=-1).values.unsqueeze(-1)
        return alphamin, alphamax

    def get_alphas_step(self, source, target):

        alphamin, alphamax = self.get_alpha_minmax(source, target)
        sdd = target - source + self.eps
        sdd_dist = torch.norm(sdd, dim=-1, keepdim=True)
        max_num_step = torch.trunc(sdd_dist.max()*(alphamax-alphamin)/self.step).max().item()+1
        alphas_step = torch.arange(max_num_step, dtype=self.dtype, device=self.device) * self.step
        alphas_step = alphas_step.expand(len(source), 1, -1)

        alphas = alphas_step / sdd_dist + alphamin

        good_idxs = torch.logical_and(alphas >= alphamin, alphas <= alphamax)
        alphas[~good_idxs] = -1
        alphas = alphas[..., ~alphas.__eq__(-1).all(dim=0).all(dim=0)]
        return alphas

    def get_index_trilinear(self, mid_alpha, source, target):
        sdd = target - source + self.eps
        idxs = (source.unsqueeze(1) + mid_alpha.unsqueeze(-1) * sdd.unsqueeze(2)) / self.spacing
        idxs[..., 0] = idxs[..., 0] / (self.dims[0] - 1) * 2.0 - 1.0
        idxs[..., 1] = idxs[..., 1] / (self.dims[1] - 1) * 2.0 - 1.0
        idxs[..., 2] = idxs[..., 2] / (self.dims[2] - 1) * 2.0 - 1.0
        idxs = idxs.unsqueeze(1)
        return idxs

    def get_voxel_trilinear(self, mid_alpha, source, target):
        idxs = self.get_index_trilinear(mid_alpha, source, target)
        idxs = idxs.permute(1, 0, 2, 3, 4)
        voxels = torch.nn.functional.grid_sample(self.volume_grid, grid=idxs, align_corners=True)
        mats = torch.nn.functional.grid_sample(self.material_grid, grid=idxs, align_corners=True)
        voxels = voxels.permute(2, 1, 0, 3, 4)
        mats = mats.permute(2, 1, 0, 3, 4)
        voxels = voxels.squeeze(2)
        mats = mats.squeeze(2)
        return voxels, mats

    def raytrace(self, source, target, target_id=None):
        alphas = self.get_alphas_step(source, target)
        alphamid = (alphas[..., 0:-1] + alphas[..., 1:]) / 2
        voxels, mats = self.get_voxel_trilinear(alphamid, source, target)
        weighted_voxels = voxels * self.step
        weighted_voxels_mats = weighted_voxels.repeat(1, self.num_materials, 1, 1)*mats
        area_density_mats = torch.nansum(weighted_voxels_mats, dim=-1) / 10.0  # step is in mm, we need to convert to cm

        beer_lambert_exp = torch.einsum("bmn,mc->bnc", area_density_mats, self.absorption_coef)
        intensity = torch.einsum("bnc,ci->bni", torch.exp(-beer_lambert_exp), (self.energy_pdf*self.energy)[:, None])
        deepdrr = intensity.squeeze(-1)
        # neg-log
        deepdrr += self.eps
        deepdrr = -torch.log(deepdrr)

        if target_id is not None:
            target_id = target_id if isinstance(target_id, list) else [target_id]
            area_density_mats_ref = area_density_mats
            for i in range(self.num_materials):
                if i not in target_id:
                    area_density_mats_ref[:, i] = 0
            beer_lambert_exp_ref = torch.einsum("bmn,mc->bnc", area_density_mats_ref, self.absorption_coef)
            intensity_ref = torch.einsum("bnc,ci->bni", torch.exp(-beer_lambert_exp_ref), (self.energy_pdf * self.energy)[:, None])
            deepdrr_ref = intensity_ref.squeeze(-1)
            # neg-log
            deepdrr_ref += self.eps
            deepdrr_ref = -torch.log(deepdrr_ref)
            return deepdrr, deepdrr_ref
        return deepdrr, None
