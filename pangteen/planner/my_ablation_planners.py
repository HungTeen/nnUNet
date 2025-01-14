from typing import Union, List, Tuple

import numpy as np

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from pangteen import config
from pangteen.network.common.helper import get_matching_instancenorm, convert_dim_to_conv_op


class MyDefault(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'default',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.batch_size = 2
        self.patch_size = config.ablation_patch_size
        self.num_stages = 6
        self.features_per_stage = [32, 64, 128, 256, 320, 320, 320, 320, 320]
        self.conv_kernel_sizes = [[3] * len(self.patch_size)] * 10
        self.strides = None


    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        unet_conv_op = convert_dim_to_conv_op(len(spacing))
        num_stages = self.num_stages

        norm = get_matching_instancenorm(unet_conv_op)
        if self.strides is None:
            strides = []
            cur_shape = list(self.patch_size)
            while len(strides) < num_stages:
                stride = []
                for dim in range(len(cur_shape)):
                    if cur_shape[dim] > self.UNet_featuremap_min_edge_length:
                        stride.append(2)
                        cur_shape[dim] = int(np.ceil(cur_shape[dim] / 2))
                    else:
                        stride.append(1)
                strides.append(stride)

            # 反转 strides
            strides = strides[::-1]
        else:
            strides = self.strides

        architecture_kwargs = {
            'network_class_name': self.UNet_class.__module__ + '.' + self.UNet_class.__name__,
            'arch_kwargs': {
                'n_stages': num_stages,
                'kernel_sizes': self.conv_kernel_sizes[:num_stages],
                'strides': strides,
                'features_per_stage': self.features_per_stage[:num_stages],
                'n_conv_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
                'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                'conv_op': unet_conv_op.__module__ + '.' + unet_conv_op.__name__,
                'conv_bias': True,
                'norm_op': norm.__module__ + '.' + norm.__name__,
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None,
                'dropout_op_kwargs': None,
                'nonlin': 'torch.nn.LeakyReLU',
                'nonlin_kwargs': {'inplace': True},
            },
            '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin'),
        }

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': self.batch_size,
            'patch_size': self.patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
            'architecture': architecture_kwargs
        }
        return plan


class MyStage5(MyDefault):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'stage5Plans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name, overwrite_target_spacing, suppress_transpose)
        self.num_stages = 5


class MyStage4(MyDefault):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'stage4Plans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name, overwrite_target_spacing, suppress_transpose)
        self.num_stages = 4


class MySpecialStage5(MyDefault):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'special5Plans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name, overwrite_target_spacing, suppress_transpose)
        self.num_stages = 5
        self.strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]