import pydoc
from typing import Union, List, Tuple

import torch
from torch import nn

from pangteen import config
from pangteen.network.medsegmamba.medsegmamba import MedSegMamba
from pangteen.network.ptnet.ptsegmamba import SFSegMamba
from pangteen.network.sbcnet.sbcnet import SBCNet
from pangteen.network.segmamba.segmamba import SegMamba
from pangteen.network.unetr.unetr import UNETR
from pangteen.trainer.trainers import HTTrainer


class SBCTrainer(HTTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False

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

        network = SBCNet(
            n_classes=3,
            in_channels=1,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network
