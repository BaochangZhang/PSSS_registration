import numpy as np
import torch
import torch.nn as nn

from libs.DiffDeepDRR.drr_projectors.proj_zbc import Deepdrrbased_Projector
from libs.DiffDeepDRR.vol.volume_Realistic import Volume_Realistc
from libs.DiffDeepDRR.device_geometry.MobileCArm import MobileCArm
from libs.DiffDeepDRR.device_geometry.utils import get_device


class Differentiable_DRRs(nn.Module):
    def __init__(
        self,
        Vol: Volume_Realistc,
        Projector: Deepdrrbased_Projector,
        height,
        pixel_size,
        width=None,
        detector_center_x=216.0, 
        detector_center_y=216.0,
        distance_source2patient=742.5,
        distance_detector2patient=517.15,
        normlized=False,
        bone_dark=True,
        Target_id=None,
        batch_size=1,
        grid_sample_step=None,
        dtype=torch.float32,
        device="cuda",
        out_dim=3,  # [b, h, w]
    ):
        super().__init__()
        self.dtype = dtype
        self.device = get_device(device)
        self.bone_dark = bone_dark

        # Initialize the X-ray detector
        width = height if width is None else width
        self.DRR_Device = MobileCArm(
            self.dtype,
            self.device,
            sensor_width=width,
            sensor_height=height,
            pixel_size=pixel_size,
            piercing_offset_x=detector_center_x,  # mm 216.0
            piercing_offset_y=detector_center_y,  # mm 277.1
            Source_Axis_Distance=distance_source2patient,
            Detector_Axis_Distance=distance_detector2patient,
        )
        self.source_position = None
        self.dectorcenter_position = None
        self.out_dim = out_dim
        volume = np.asarray(Vol.get_data())
        spacing = list(Vol.get_spacing())
        volume_size = volume.shape
        volume_center = (np.asarray(volume_size)-0) * np.asarray(spacing) / 2.0

        print('volume center:', volume_center)
        self.DRR_Device.reposition(list(volume_center))

        self.proj = Projector

        self.register_parameter("rotations", None)
        self.register_parameter("translations", None)

        self.InstanceNorm = nn.InstanceNorm1d(batch_size)
        self.normalized = normlized
        self.Target_id = Target_id
        self.grid_sampel_step = None
        if grid_sample_step is not None:
            assert int(grid_sample_step) > 0, f"Warning: the grid_sample_step must be int or not zero!"
            self.grid_sampel_step = grid_sample_step

    def forward(
        self,
        alpha=None,
        beta=None,
        gamma=None,
        bx=None,
        by=None,
        bz=None,
        batch=None,
    ):
        """
        Generate a DRR from a particular viewing angle.
        Pass projector parameters to initialize a new viewing angle.
        If uninitialized, the model will not run.
        """
        if batch is not None:
            rotations = batch[..., 0:3]
            translations = batch[..., 3:]
            source, target, center = self.DRR_Device.make_xrays(rotations, translations, self.grid_sampel_step)
            self.source_position = source
            self.dectorcenter_position = center
        else:
            params = [alpha, beta, gamma, bx, by, bz]
            if any(arg is not None for arg in params):
                self.initialize_parameters(*params)
            source, target, center = self.DRR_Device.make_xrays(
                self.rotations,
                self.translations,
                self.grid_sampel_step,
            )
            self.source_position = source
            self.dectorcenter_position = center

        img, img_ref = self.proj.raytrace(source, target, target_id=self.Target_id)

        if self.normalized:
            img = self.InstanceNorm(img)
            img = torch.sigmoid(img)

        img = (img - img.min(1, keepdim=True)[0]) / (img.max(1, keepdim=True)[0] - img.min(1, keepdim=True)[0])
        if self.bone_dark:
            img = 1.0 - img

        if self.grid_sampel_step is not None:
            out = img.view(-1, (self.DRR_Device.height - 1) // self.grid_sampel_step + 1,
                           (self.DRR_Device.width - 1) // self.grid_sampel_step + 1)
        else:
            out = img.view(-1, self.DRR_Device.height, self.DRR_Device.width)

        # with torch.autograd.detect_anomaly():
        #     out = equalize_clahe(out, clip_limit=1.0, slow_and_differentiable=True)

        if self.out_dim == 4:
            out = out.unsqueeze(1)

        if img_ref is not None:
            img_ref = (img_ref - img_ref.min(1, keepdim=True)[0]) / (
                        img_ref.max(1, keepdim=True)[0] - img_ref.min(1, keepdim=True)[0])
            if self.bone_dark:
                img_ref = 1.0 - img_ref
            if self.grid_sampel_step is not None:
                out_ref = img_ref.view(-1, (self.DRR_Device.height - 1) // self.grid_sampel_step + 1,
                                       (self.DRR_Device.width - 1) // self.grid_sampel_step + 1)
            else:
                out_ref = img_ref.view(-1, self.DRR_Device.height, self.DRR_Device.width)
            if self.out_dim == 4:
                out_ref = out_ref.unsqueeze(1)

            return out, out_ref

        return out, None

    def initialize_parameters(self, alpha, beta, gamma, bx, by, bz):
        tensor_args = {"dtype": self.dtype, "device": self.device}
        # Assume that C-arm geometry is given for a 6DoF registration problem
        self.rotations = nn.Parameter(
            torch.tensor([[alpha, beta, gamma]], **tensor_args))
        self.translations = nn.Parameter(torch.tensor([[bx, by, bz]], **tensor_args))

    def __repr__(self):
        params = [str(param) for param in self.parameters()]
        if len(params) == 0:
            return "Parameters uninitialized."
        else:
            return "\n".join(params)
