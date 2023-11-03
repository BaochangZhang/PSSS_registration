
import torch
from torch.nn.functional import normalize
from .utils import get_device
import numpy as np
# np.set_printoptions(suppress=True)


class MobileCArm:
    def __init__(self, dtype, device,
                 Source_Axis_Distance: float = 742.5,
                 Detector_Axis_Distance: float = 517.15,
                 piercing_offset_x: float = 216.0,
                 piercing_offset_y: float = 216.1,
                 sensor_height=1440,
                 sensor_width=1440,
                 pixel_size=0.3,
                 isocenter=[0, 0, 0],):

        self.Source_Axis_Distance = Source_Axis_Distance
        self.Detector_Axis_Distance = Detector_Axis_Distance

        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.pixel_size = pixel_size

        self.source_isocenter_offset_x = piercing_offset_x - sensor_width / 2 * pixel_size
        self.source_isocenter_offset_y = piercing_offset_y - sensor_height / 2 * pixel_size

        self.isocenter = isocenter

        self.height = sensor_height
        self.width = sensor_width
        self.delx = pixel_size
        self.dely = pixel_size

        self.dtype = dtype
        self.device = device if isinstance(device, torch.device) else get_device(device)

    def make_xrays(self, rotations, translations, grid_sample_step=None):
        """
        Inputs
        ------
        sdr : torch.Tensor
            Source-to-Detector radius (half of the Source-to-Detector distance)
        rotations : torch.Tensor
            Vector of C-Arm rotations (theta, phi, gamma) for azimuthal, polar, and roll
        translations : torch.Tensor
            Vector of volume translations (bx, by, bz)

        # it would be great to do a sample on image, which could be used to set a loss.
        """

        # Get the detector plane normal vector
        assert len(rotations) == len(translations)
        source, center, basis = self.get_basis(rotations, self.device)
        source += translations.unsqueeze(1)
        center += translations.unsqueeze(1)
        # Construct the detector plane
        t = (torch.arange(0, self.height)-(self.height-1)/2) * self.delx
        s = (torch.arange(0, self.width)-(self.width-1)/2) * self.dely

        coefs = torch.cartesian_prod(t, s).reshape(-1, 2).to(self.device).to(self.dtype)
        target = torch.einsum("bcd,nc->bnd", basis, coefs)
        target += center  # [batch, h*w, 3]
        # grid-sample
        if grid_sample_step is not None:
            target_2d = target.view(-1, self.height, self.width, 3)
            target_2d = target_2d[:, ::grid_sample_step, ::grid_sample_step, :]
            target = target_2d.reshape(-1, ((self.height-1)//grid_sample_step+1)*((self.width-1)//grid_sample_step+1), 3)
        return source, target, center

    def Rx(self, beta, batch_size, device):
        beta = beta / 180.0 * torch.pi
        t0 = torch.zeros(batch_size, 1, device=device)
        t1 = torch.ones(batch_size, 1, device=device)
        return torch.stack(
            [
                t1, t0, t0, t0,
                t0, torch.cos(beta.unsqueeze(1)), -torch.sin(beta.unsqueeze(1)), t0,
                t0, torch.sin(beta.unsqueeze(1)), torch.cos(beta.unsqueeze(1)), t0,
                t0, t0, t0, t1
            ],
            dim=1,
        ).reshape(batch_size, 4, 4)

    def Ry(self, alpha, batch_size, device):
        alpha = alpha / 180.0 * torch.pi
        t0 = torch.zeros(batch_size, 1, device=device)
        t1 = torch.ones(batch_size, 1, device=device)
        return torch.stack(
            [
                torch.cos(alpha.unsqueeze(1)),  t0,  torch.sin(alpha.unsqueeze(1)),  t0,
                t0, t1, t0, t0,
                -torch.sin(alpha.unsqueeze(1)), t0, torch.cos(alpha.unsqueeze(1)),  t0,
                t0, t0, t0, t1,
            ],
            dim=1,
        ).reshape(batch_size, 4, 4)

    def Rz(self, gamma, batch_size, device):
        gamma = gamma / 180.0 * torch.pi
        t0 = torch.zeros(batch_size, 1, device=device)
        t1 = torch.ones(batch_size, 1, device=device)
        return torch.stack(
            [
                torch.cos(gamma.unsqueeze(1)),  -torch.sin(gamma.unsqueeze(1)), t0, t0,
                torch.sin(gamma.unsqueeze(1)),  torch.cos(gamma.unsqueeze(1)),  t0, t0,
                t0, t0, t1, t0,
                t0, t0, t0, t1
            ],
            dim=1,
        ).reshape(batch_size, 4, 4)

    def reposition(self, new_isocenter):
        vol_isocenter_offset_x = self.source_isocenter_offset_x * self.Source_Axis_Distance / (
                    self.Source_Axis_Distance + self.Detector_Axis_Distance)
        vol_isocenter_offset_y = self.source_isocenter_offset_y * self.Source_Axis_Distance / (
                    self.Source_Axis_Distance + self.Detector_Axis_Distance)
        new_isocenter[0] = new_isocenter[0] + vol_isocenter_offset_x
        new_isocenter[1] = new_isocenter[1] + vol_isocenter_offset_y
        self.isocenter = new_isocenter

    def from_trans(self, trans, batch_size, device):
        pos = torch.Tensor(trans).to(device=device)
        trans = torch.eye(4, 4, device=device)
        trans[0:3, 3] = pos
        return trans.repeat(batch_size, 1, 1)

    def Source_Rt(self, rotations, device):
        alpha, beta, gamma = rotations[:, 0], rotations[:, 1], rotations[:, 2]
        batch_size = len(rotations)
        R_x = self.Rx(beta, batch_size, device)
        R_y = self.Ry(alpha, batch_size, device)
        R_z = self.Rz(gamma, batch_size, device)
        T_s = self.from_trans(trans=[0, 0, self.Source_Axis_Distance], batch_size=batch_size, device=device)
        T_isocenter = self.from_trans(trans=self.isocenter, batch_size=batch_size, device=device)
        world_from_source = torch.einsum("bni,bij,bjk,bkl,blm->bnm", T_isocenter, R_z, R_x, R_y, T_s)
        return world_from_source

    def Detector_Rt(self, rotations, device):
        alpha, beta, gamma = rotations[:, 0], rotations[:, 1], rotations[:, 2]
        # if detector is always on the opposite of source, need to more things
        batch_size = len(rotations)
        R_x = self.Rx(beta, batch_size, device)
        R_y = self.Ry(alpha, batch_size, device)
        R_z = self.Rz(gamma, batch_size, device)
        T_s = self.from_trans(trans=[-self.source_isocenter_offset_x,
                                     -self.source_isocenter_offset_y,
                                     -self.Detector_Axis_Distance], batch_size=batch_size, device=device)
        T_isocenter = self.from_trans(trans=self.isocenter, batch_size=batch_size, device=device)
        world_from_detector = torch.einsum("bni,bij,bjk,bkl,blm->bnm", T_isocenter, R_z, R_x, R_y, T_s)
        return world_from_detector

    def get_basis(self, rotations, device):
        # Get the rotation of 3D space
        sRt = self.Source_Rt(rotations, device)
        dRt = self.Detector_Rt(rotations, device)
        # Get the detector center and X-ray source
        source = sRt[:, 0:3, 3].unsqueeze(1)
        center = dRt[:, 0:3, 3].unsqueeze(1)
        # Get the basis of the detector plane (before translation)
        R_ = normalize(dRt[:, 0:3, 0:3].clone(), dim=-1)
        u, v = R_[..., 0], R_[..., 1]

        basis = torch.stack([v, -u], dim=1)
        return source, center, basis