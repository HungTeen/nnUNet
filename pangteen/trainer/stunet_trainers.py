from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from torch import nn

from pangteen.network.stunet.stunet import STUNet


class STUNetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-2

    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        return STUNet(num_input_channels, num_classes, depth=[1] * 6, dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                      enable_deep_supervision=enable_deep_supervision)


class STUNetTrainer_small(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        return STUNet(num_input_channels, num_classes, depth=[1] * 6, dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                      enable_deep_supervision=enable_deep_supervision)


class STUNetTrainer_small_ft(STUNetTrainer_small):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000


class STUNetTrainer_base(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        return STUNet(num_input_channels, num_classes, depth=[1] * 6, dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                      enable_deep_supervision=enable_deep_supervision)


class STUNetTrainer_base_ft(STUNetTrainer_base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000


class STUNetTrainer_large(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        return STUNet(num_input_channels, num_classes, depth=[2] * 6, dims=[64 * x for x in [1, 2, 4, 8, 16, 16]],
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                      enable_deep_supervision=enable_deep_supervision)


class STUNetTrainer_large_ft(STUNetTrainer_large):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000


class STUNetTrainer_huge(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        return STUNet(num_input_channels, num_classes, depth=[3] * 6, dims=[96 * x for x in [1, 2, 4, 8, 16, 16]],
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                      enable_deep_supervision=enable_deep_supervision)


class STUNetTrainer_huge_ft(STUNetTrainer_huge):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000

