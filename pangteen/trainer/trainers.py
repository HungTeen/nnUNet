from os.path import join, isfile
from typing import Union, List, Tuple, Optional

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert3DTo2DTransform
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.crossval_split import generate_lock_split
from pangteen import config
from pangteen.network.common.helper import get_matching_dropout


class HTTrainer(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300

    # def do_split(self):
    #     """
    #     固定划分训练集和验证集。
    #     """
    #     if self.fold == "all":
    #         # if fold==all then we use all images for training and validation
    #         case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
    #         tr_keys = case_identifiers
    #         val_keys = tr_keys
    #     else:
    #         splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
    #         dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
    #                                 num_images_properties_loading_threshold=0,
    #                                 folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
    #         # if the split file does not exist we need to create it
    #         if not isfile(splits_file):
    #             self.print_to_log_file("Creating new 5-fold cross-validation split...")
    #             all_keys_sorted = list(np.sort(list(dataset.keys())))
    #             splits = generate_lock_split(all_keys_sorted, seed=12345, n_splits=5)
    #             save_json(splits, splits_file)
    #
    #         else:
    #             self.print_to_log_file("Using splits from existing split file:", splits_file)
    #             splits = load_json(splits_file)
    #             self.print_to_log_file(f"The split file contains {len(splits)} splits.")
    #
    #         self.print_to_log_file("Desired fold for training: %d" % self.fold)
    #         if self.fold < len(splits):
    #             tr_keys = splits[self.fold]['train']
    #             val_keys = splits[self.fold]['val']
    #             self.print_to_log_file("This split has %d training and %d validation cases."
    #                                    % (len(tr_keys), len(val_keys)))
    #         else:
    #             self.print_to_log_file("INFO: You requested fold %d for training but splits "
    #                                    "contain only %d folds. I am now creating a "
    #                                    "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
    #             # if we request a fold that is not in the split file, create a random 80:20 split
    #             rnd = np.random.RandomState(seed=12345 + self.fold)
    #             keys = np.sort(list(dataset.keys()))
    #             idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
    #             idx_val = [i for i in range(len(keys)) if i not in idx_tr]
    #             tr_keys = [keys[i] for i in idx_tr]
    #             val_keys = [keys[i] for i in idx_val]
    #             self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
    #                                    % (len(tr_keys), len(val_keys)))
    #         if any([i in val_keys for i in tr_keys]):
    #             self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
    #                                    'splits.json or ignore if this is intentional.')
    #     return tr_keys, val_keys

    # @staticmethod
    # def get_training_transforms(
    #         patch_size: Union[np.ndarray, Tuple[int]],
    #         rotation_for_DA: RandomScalar,
    #         deep_supervision_scales: Union[List, Tuple, None],
    #         mirror_axes: Tuple[int, ...],
    #         do_dummy_2d_data_aug: bool,
    #         use_mask_for_norm: List[bool] = None,
    #         is_cascaded: bool = False,
    #         foreground_labels: Union[Tuple[int, ...], List[int]] = None,
    #         regions: List[Union[List[int], Tuple[int, ...], int]] = None,
    #         ignore_label: int = None,
    # ) -> BasicTransform:
    #     transforms = []
    #     if do_dummy_2d_data_aug:
    #         ignore_axes = (0,)
    #         transforms.append(Convert3DTo2DTransform())
    #         patch_size_spatial = patch_size[1:]
    #     else:
    #         patch_size_spatial = patch_size
    #         ignore_axes = None
    #
    #     rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    #     transforms.append(
    #         SpatialTransform(
    #             patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
    #             p_rotation=0.2,
    #             rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.85, 1.15), p_synchronize_scaling_across_axes=1,
    #             # rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
    #             bg_style_seg_sampling=False  # , mode_seg='nearest'
    #         )
    #     )
    #
    #     if do_dummy_2d_data_aug:
    #         transforms.append(Convert2DTo3DTransform())
    #
    #     transforms.append(RandomTransform(
    #         GaussianNoiseTransform(
    #             noise_variance=(0, 0.1),
    #             p_per_channel=1,
    #             synchronize_channels=True
    #         ), apply_probability=0.1
    #     ))
    #     transforms.append(RandomTransform(
    #         GaussianBlurTransform(
    #             blur_sigma=(0.5, 1.),
    #             synchronize_channels=False,
    #             synchronize_axes=False,
    #             p_per_channel=0.5, benchmark=True
    #         ), apply_probability=0.2
    #     ))
    #     transforms.append(RandomTransform(
    #         MultiplicativeBrightnessTransform(
    #             multiplier_range=BGContrast((0.75, 1.25)),
    #             synchronize_channels=False,
    #             p_per_channel=1
    #         ), apply_probability=0.15
    #     ))
    #     transforms.append(RandomTransform(
    #         ContrastTransform(
    #             contrast_range=BGContrast((0.75, 1.25)),
    #             preserve_range=True,
    #             synchronize_channels=False,
    #             p_per_channel=1
    #         ), apply_probability=0.15
    #     ))
    #     transforms.append(RandomTransform(
    #         SimulateLowResolutionTransform(
    #             scale=(0.5, 1),
    #             synchronize_channels=False,
    #             synchronize_axes=True,
    #             ignore_axes=ignore_axes,
    #             allowed_channels=None,
    #             p_per_channel=0.5
    #         ), apply_probability=0.25
    #     ))
    #     transforms.append(RandomTransform(
    #         GammaTransform(
    #             gamma=BGContrast((0.7, 1.5)),
    #             p_invert_image=1,
    #             synchronize_channels=False,
    #             p_per_channel=1,
    #             p_retain_stats=1
    #         ), apply_probability=0.1
    #     ))
    #     transforms.append(RandomTransform(
    #         GammaTransform(
    #             gamma=BGContrast((0.7, 1.5)),
    #             p_invert_image=0,
    #             synchronize_channels=False,
    #             p_per_channel=1,
    #             p_retain_stats=1
    #         ), apply_probability=0.3
    #     ))
    #     # 取消镜像。
    #     # if mirror_axes is not None and len(mirror_axes) > 0:
    #     #     transforms.append(
    #     #         MirrorTransform(
    #     #             allowed_axes=mirror_axes
    #     #         )
    #     #     )
    #
    #     if use_mask_for_norm is not None and any(use_mask_for_norm):
    #         transforms.append(MaskImageTransform(
    #             apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
    #             channel_idx_in_seg=0,
    #             set_outside_to=0,
    #         ))
    #
    #     transforms.append(
    #         RemoveLabelTansform(-1, 0)
    #     )
    #
    #     if regions is not None:
    #         # the ignore label must also be converted
    #         transforms.append(
    #             ConvertSegmentationToRegionsTransform(
    #                 regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
    #                 channel_in_seg=0
    #             )
    #         )
    #
    #     if deep_supervision_scales is not None:
    #         transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
    #
    #     return ComposeTransforms(transforms)


    @staticmethod
    def update_network_args(arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool,
                                   invalid_args: list[str] = None,
                                   n_stages: Optional[int] = None,
                                   print_args: bool = False) -> dict:
        """
        构建模型的通用部分。
        """
        from pydoc import locate
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = locate(architecture_kwargs[ri])

        if enable_deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = enable_deep_supervision

        architecture_kwargs['input_channels'] = num_input_channels
        architecture_kwargs['num_classes'] = num_output_channels

        if invalid_args:
            for i in invalid_args:
                architecture_kwargs.pop(i)

        if n_stages:
            architecture_kwargs['n_stages'] = n_stages
            architecture_kwargs['features_per_stage'] = architecture_kwargs['features_per_stage'][:n_stages]
            architecture_kwargs['kernel_sizes'] = architecture_kwargs['kernel_sizes'][:n_stages]
            architecture_kwargs['strides'] = architecture_kwargs['strides'][:n_stages]
            architecture_kwargs['n_conv_per_stage'] = architecture_kwargs['n_conv_per_stage'][:n_stages]
            architecture_kwargs['n_conv_per_stage_decoder'] = architecture_kwargs['n_conv_per_stage_decoder'][:n_stages - 1]

        if config.drop_out_rate:
            architecture_kwargs['dropout_op'] = get_matching_dropout(dimension=3)
            architecture_kwargs['dropout_op_kwargs'] = {'p': config.drop_out_rate}

        if print_args:
            print("Look ! It's architecture args : {}", architecture_kwargs)

        return architecture_kwargs


    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        让模型选择变成继承制的，不用每次去改 Plan。
        """
        architecture_kwargs = HTTrainer.update_network_args(arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels, num_output_channels, enable_deep_supervision)

        network = PlainConvUNet(**architecture_kwargs)

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network