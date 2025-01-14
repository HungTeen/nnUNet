import pydoc
from typing import Union, List, Tuple

import torch
from monai.networks.nets import SegResNet
from torch import nn

from pangteen import config
from pangteen.network.lightm_unet.lightm_unet import LightMUNet
from pangteen.network.segmamba.segmamba import SegMamba
from pangteen.network.unetr.unetr import UNETR
from pangteen.trainer.trainers import HTTrainer


class SegResNetTrainer(HTTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
        self.initial_lr = 1e-4
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        network = SegResNet(
            spatial_dims = 3,
            init_filters = 32,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
        )


        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network