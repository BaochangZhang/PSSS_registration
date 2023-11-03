import os
import glob
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from decimal import Decimal
from .spectral_data import spectrums
from .material_coefficients import material_coefficients


class Volume_Realistc(object):
    def __init__(
            self,
            cache_path: Path,
            data: np.ndarray,
            spacing: list,
            materials: dict,
            energies: np.ndarray,
            energies_pdf: np.ndarray,
            orientation: str):

        self.__cache_path = cache_path
        self.__numpydata = np.array(data).astype(np.float32)
        self.__spacing = spacing
        self.__orientation = orientation
        self.__materials = materials
        self.__spectrum_kev = energies
        self.__spectrum_pdf = energies_pdf  # Probability density functions

        # These will be initialized after run Update()
        self.__materials_backup = self.__materials  # material backup
        self.__absorption_coef = None  # np.ndarray
        self.__mats = None  # np.ndarray
        self.__Update_check = False

    @classmethod
    def from_nifti(
            cls,
            filepath: str,
            HU_segments: list,
            target_orient='LPS',
            spectrum="90KV_AL40",  # '60KV_AL35','90KV_AL40', '120KV_AL43'
            spectrum_downsample=2,
            resample=False,
            resample_ratio=2.0,
            resample_spacing=None,
            auto_cache=False,
            use_cache=False,
            cache_dir=None,
            **kwargs,):

        if cache_dir is None:
            cache_dir = str(filepath).replace('.nii', '_'+target_orient+'_realistic.cache.zbc.npz')

        if use_cache and Path(cache_dir).exists():
            data = np.load(str(cache_dir), allow_pickle=True)
            volume = np.asarray(data['vol'])
            spacing = list(data['spacing'])
            materials = data['materials'].item()
            energies = data['spectrum_kev']
            energies_pdf = data['spectrum_pdf']
            return cls(
                cache_path=Path(cache_dir),
                data=volume,
                spacing=spacing,
                materials=materials,
                energies=energies,
                energies_pdf=energies_pdf,
                orientation=target_orient,
                **kwargs,
            )

        print('processing', filepath)
        sitk_vol = sitk.ReadImage(str(filepath))
        origin_orient = cls.Get_Orient(sitk_vol.GetDirection())
        print('the orientation of raw image is ', origin_orient)
        print('the spacing of raw image is ', sitk_vol.GetSpacing())

        if not cls.orientation_valid(target_orient):
            target_orient = origin_orient
            print('Nothing is done, keep orientation as ', target_orient)

        if origin_orient != target_orient:
            print('Change the orientation to ', target_orient)
            sitk_vol = sitk.DICOMOrient(sitk_vol, target_orient)
        if resample:
            if resample_spacing is None:
                temp_spacing = list(np.asarray(sitk_vol.GetSpacing()) * resample_ratio)
            else:
                temp_spacing = resample_spacing
            # sitk_vol_resample, spacing = cls.resample_volume(sitk_vol, temp_spacing=temp_spacing, interpolator=sitk.sitkLinear)
            sitk_vol_resample, spacing = cls.resample_volume(sitk_vol, temp_spacing=temp_spacing, interpolator=sitk.sitkBSpline)
            # simpleitk to numpy
            volume = sitk.GetArrayFromImage(sitk_vol_resample).transpose(2, 1, 0)  # ijk==lps

        else:
            spacing = list(np.asarray(sitk_vol.GetSpacing()) * 1.0)
            # simpleitk to numpy
            volume = sitk.GetArrayFromImage(sitk_vol).transpose(2, 1, 0)  # ijk==lps

        # get basic CT materials segmentation: Air/Soft-tissue/Bone
        seg_materials = cls.conv_hu_to_materials(volume, HU_segments)

        spectrum_data = cls.get_spectrum(spectrum, spectrum_downsample)
        energies = spectrum_data[:, 0].copy() / 1000
        contiguous_energies = np.ascontiguousarray(energies, dtype=np.float32)
        # n_bins = contiguous_energies.shape[0]

        pdf_energy = spectrum_data[:, 1] / np.sum(spectrum_data[:, 1])
        contiguous_pdf = np.ascontiguousarray(pdf_energy.copy(), dtype=np.float32)

        density = cls.conv_hu_to_density(volume)

        # save cache
        if auto_cache:
            np.savez(cache_dir, vol=density, spacing=spacing, materials=seg_materials, spectrum_kev=contiguous_energies,
                     spectrum_pdf=contiguous_pdf)
            print('cache is saved in ', cache_dir)

        return cls(
            cache_path=Path(cache_dir),
            data=density,
            spacing=list(spacing),
            materials=seg_materials,
            energies=contiguous_energies,
            energies_pdf=contiguous_pdf,
            orientation=target_orient,
            **kwargs,
        )

    @staticmethod
    def get_absorption_coefs(energy_keV, material):
        """Returns the absorption coefficient for the specified material at the specified energy level (in keV)

        Args:
            energy_keV: energy level of photon/ray (keV)
            material (str): the material

        Returns:
            the absorption coefficient (in [cm^2 / g]), interpolated from the data in material_coefficients.py
        """

        def log_interp(xInterp, x, y):
            # xInterp is the single energy value to interpolate an absorption coefficient for,
            # interpolating from the data from "x" (energy value array from slicing material_coefficients)
            # and from "y" (absorption coefficient array from slicing material_coefficients)
            xInterp = np.log10(xInterp.copy())
            x = np.log10(x.copy())
            y = np.log10(y.copy())
            yInterp = np.power(10, np.interp(xInterp, x, y))  # np.interp is 1-D linear interpolation
            return yInterp

        xMev = energy_keV / 1000
        return log_interp(xMev, material_coefficients[material][:, 0], material_coefficients[material][:, 1])

    @staticmethod
    def get_spectrum(spectrum, step):
        """Get the data corresponding to the given spectrum name.
        """
        if isinstance(spectrum, str):
            if spectrum not in spectrums:
                raise KeyError(f"unrecognized spectrum: {spectrum}")
            return spectrums[spectrum][::step, :]
        else:
            raise TypeError(f"unrecognized spectrum type: {type(spectrum)}")

    @staticmethod
    def Get_Orient(Direction):
        Orient_Matrix = np.asmatrix(Direction).reshape(3, 3)
        Orient_dict = {1: 'L', 2: 'P', 3: 'S',
                       -1: 'R', -2: 'A', -3: 'I'}
        orientKey = np.squeeze(np.asarray(Orient_Matrix.T.dot(np.array([1, 2, 3])), dtype=np.int8))
        Orient = Orient_dict[orientKey[0]] + Orient_dict[orientKey[1]] + Orient_dict[orientKey[2]]
        return Orient

    @staticmethod
    def resample_volume(sitk_volume, temp_spacing, interpolator):
        original_spacing = sitk_volume.GetSpacing()
        original_size = sitk_volume.GetSize()
        original_center = (np.asarray(original_size)) * np.asarray(original_spacing) / 2.0

        new_size = [int(Decimal(str(osz * ospc / nspc)).quantize(Decimal("1"), rounding="ROUND_HALF_UP"))
                    for osz, ospc, nspc in zip(original_size, original_spacing, temp_spacing)]

        new_spacing = original_center * 2.0 / np.asarray(new_size)

        return sitk.Resample(sitk_volume, new_size, sitk.Transform(), interpolator,
                             sitk_volume.GetOrigin(), new_spacing, sitk_volume.GetDirection(), 0,
                             sitk_volume.GetPixelID()), new_spacing

    @staticmethod
    def orientation_valid(ori):
        ori_list = ['LIP', 'RIA', 'LSA', 'PIR', 'AIL', 'ASR', 'PSL']
        if ori in ori_list:
            return True
        else:
            print('The target_axcode is not valid, Please reference ', ori_list)

            return False

    @staticmethod
    def conv_hu_to_density(hu_values, smoothAir=True):
        # Use two linear interpolations from data: (HU,g/cm^3)
        # use for lower HU: density = 0.001029*HU + 1.03
        # use for upper HU: density = 0.0005886*HU + 1.03

        # set air densities
        if smoothAir:
            hu_values[hu_values <= -1000] = -1000
        densities = np.maximum(
            np.minimum(0.001030 * hu_values + 1.030, 0.0005886 * hu_values + 1.030), 0.0)
        return densities

    @staticmethod
    def conv_hu_to_materials(hu_values, HU_segments=None):
        # this part can be improved based on pre-trained segmentation network,
        # currently it is based on threshold segmentation
        # here just provide basic material label. no including anatomy label
        materials = {}
        if HU_segments is None:
            HU_segments = [-800, 350]
        # for CT
        materials["air"] = hu_values <= HU_segments[0]
        materials["soft tissue"] = (HU_segments[0] < hu_values) * (hu_values <= HU_segments[1])  #350
        materials["bone"] = HU_segments[1] < hu_values
        return materials

    def __adjust_mask_space(self, maskpath):
        sitk_mask = sitk.ReadImage(str(maskpath))
        origin_orient = self.Get_Orient(sitk_mask.GetDirection())
        if origin_orient != self.__orientation:
            sitk_mask = sitk.DICOMOrient(sitk_mask, self.__orientation)
        temp_spacing = self.__spacing
        if not (list(temp_spacing) == list(np.asarray(sitk_mask.GetSpacing()))):
            sitk_mask, _ = self.resample_volume(sitk_mask, temp_spacing=temp_spacing,
                                                interpolator=sitk.sitkNearestNeighbor)
        mask_npdata = sitk.GetArrayFromImage(sitk_mask).transpose(2, 1, 0)  # ijk==lps

        return mask_npdata

    def Additional_mask(self, maskpath, materialName, anatomyName):
        mask = self.__adjust_mask_space(maskpath)
        mask = (0 < mask).astype(np.float32)

        TargetName = materialName
        if anatomyName is not None:
            TargetName = TargetName + '_' + str(anatomyName)

        existing_objs = self.__materials.keys()
        assert materialName in material_coefficients, f"unrecognized material: {materialName}"
        assert TargetName not in existing_objs, f"already existed object: {TargetName}"

        for obj in existing_objs:
            self.__materials[obj] = self.__materials[obj] * (1.0 - mask)
        self.__materials[TargetName] = mask
        self.__materials_backup = self.__materials
        self.__Update_check = False

    def Update(self):
        volume = self.__numpydata
        all_objs = self.__materials.keys()
        materials_mask = np.zeros((len(all_objs), volume.shape[0], volume.shape[1], volume.shape[2])).astype(np.float32)
        for i, obj in enumerate(all_objs):
            assert str(obj).split('_')[0] in material_coefficients, f"unrecognized material: {obj}"
            materials_mask[i] = self.__materials[obj]
        self.__mats = materials_mask
        n_bins = self.__spectrum_kev.shape[0]
        absorption_coef_table = np.empty((len(all_objs), n_bins), dtype=np.float32)
        for i, obj_name in enumerate(all_objs):
            temp_table = np.empty(n_bins, dtype=np.float32)
            mat_name = str(obj_name).split('_')[0]
            for bin in range(n_bins):
                temp_table[bin] = self.get_absorption_coefs(self.__spectrum_kev[bin], mat_name)
            absorption_coef_table[i] = temp_table
        self.__absorption_coef = absorption_coef_table
        self.__Update_check = True

    def get_target_objectID(self, objname):
        all_objs = self.__materials.keys()
        assert objname in all_objs, f"unrecognized object: {objname}"
        return list(all_objs).index(objname)

    def disable_materials(self, materialName):
        for mat in materialName if isinstance(materialName, list) else [materialName]:
            assert mat in self.__materials.keys(), f"unrecognized material: {mat}"
            del self.__materials[mat]
        self.__Update_check = False

    def enable_material(self, materialName):
        assert materialName in self.__materials_backup.keys(), f"unrecognized material in backup: {materialName}"
        self.__materials[materialName] = self.__materials_backup[materialName]
        self.__Update_check = False

    def get_volume_path(self):
        return self.__cache_path

    def get_data(self):
        return self.__numpydata

    def get_spacing(self):
        return self.__spacing

    def get_orientation(self):
        return self.__orientation

    def get_materials_dictionary(self):
        return self.__materials

    def get_materials(self):
        return self.__mats

    def get_spectrum_kev(self):
        return self.__spectrum_kev

    def get_spectrum_pdf(self):
        return self.__spectrum_pdf

    def get_absorption_coef(self):
        return self.__absorption_coef

    def check_ready(self):
        return self.__Update_check