from typing import Union, List, Tuple

import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

from pangteen.network.ptnet.ptunet import PangTeenUNet
from pangteen.trainer.trainers import HTTrainer


class UNetTrainer(HTTrainer):
    """
    CUDA_VISIBLE_DEVICES=2 nohup python -u pangteen/train.py 201 3d_fullres 3 -p default -tr UNetTrainer -num_gpus 1 > main.out 2>&1 &
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = HTTrainer.update_network_args(arch_init_kwargs, arch_init_kwargs_req_import,
                                                            num_input_channels, num_output_channels,
                                                            enable_deep_supervision,
                                                            print_args=True)

        network = PlainConvUNet(**architecture_kwargs)

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class PTUNetTrainer(HTTrainer):
    """
    CUDA_VISIBLE_DEVICES=2 nohup python -u pangteen/train.py 201 3d_fullres 3 -p default -tr PTUNetTrainer -num_gpus 1 > main.out 2>&1 &
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = HTTrainer.update_network_args(arch_init_kwargs, arch_init_kwargs_req_import,
                                                            num_input_channels, num_output_channels,
                                                            enable_deep_supervision,
                                                            print_args=True)

        network = PangTeenUNet(**architecture_kwargs)

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class SmallUNetTrainer(HTTrainer):
    """
    CUDA_VISIBLE_DEVICES=2 nohup python -u pangteen/train.py 201 3d_fullres 3 -p default -tr SmallUNetTrainer -num_gpus 1 > main.out 2>&1 &
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 600

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = HTTrainer.update_network_args(arch_init_kwargs, arch_init_kwargs_req_import,
                                                            num_input_channels, num_output_channels,
                                                            enable_deep_supervision,
                                                            print_args=True)

        initial_feature_size = 16
        architecture_kwargs['features_per_stage'] = [initial_feature_size * 2 ** i for i in range(architecture_kwargs['n_stages'])]

        network = PlainConvUNet(**architecture_kwargs)

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network