from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union, Sequence

import torch
from torch import nn, Tensor

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
from torchvision.utils import _make_ntuple

import freezable_modules.modules as mod

########################################################################################################################
###################################### file derived from torchvision source code #######################################
########################################################################################################################


class FreezableConv2dNormActivation(mod.SequentialF):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            mod.FreezableConv2d(
                in_channels,
                out_channels,
                kernel_size,
                kernel_size,
                bias=bias,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=groups,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        _log_api_usage_once(self)
        self.out_channels = out_channels

# necessary for backwards compatibility
class FreezableInvertedResidual(mod.FreezableModule):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None, conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if conv_layer is None:
            conv_layer = mod.FreezableConv2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                FreezableConv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                FreezableConv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                mod.FreezableConv2d(hidden_dim, oup, 1, 1, stride=1, padding=0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = mod.SequentialF(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1
        self.set_freezing_matrix(self.get_default_freezing_matrix())

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(mod.FreezableModule):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()
        _log_api_usage_once(self)

        if block is None:
            block = FreezableInvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        elif inverted_residual_setting == "c10" or inverted_residual_setting == "c100":
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],
                [6, 32, 3, 1],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            FreezableConv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            FreezableConv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = mod.SequentialF(*features)

        # building classifier
        self.classifier = mod.SequentialF(
            nn.Dropout(p=dropout),
            mod.FreezableLinear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, mod.FreezableConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, mod.FreezableLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.set_freezing_matrix(self.get_default_freezing_matrix())

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

if True: #TODO

    _COMMON_META = {
        "num_params": 3504872,
        "min_size": (1, 1),
        "categories": _IMAGENET_CATEGORIES,
    }


    class MobileNet_V2_Weights(WeightsEnum):
        IMAGENET1K_V1 = Weights(
            url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
            transforms=partial(ImageClassification, crop_size=224),
            meta={
                **_COMMON_META,
                "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 71.878,
                        "acc@5": 90.286,
                    }
                },
                "_ops": 0.301,
                "_file_size": 13.555,
                "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
            },
        )
        IMAGENET1K_V2 = Weights(
            url="https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
            transforms=partial(ImageClassification, crop_size=224, resize_size=232),
            meta={
                **_COMMON_META,
                "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 72.154,
                        "acc@5": 90.822,
                    }
                },
                "_ops": 0.301,
                "_file_size": 13.598,
                "_docs": """
                    These weights improve upon the results of the original paper by using a modified version of TorchVision's
                    `new training recipe
                    <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
                """,
            },
        )
        DEFAULT = IMAGENET1K_V2


    @register_model()
    @handle_legacy_interface(weights=("pretrained", MobileNet_V2_Weights.IMAGENET1K_V1))
    def freezablemobilenet_v2(
        *, weights: Optional[MobileNet_V2_Weights] = None, progress: bool = True, **kwargs: Any
    ) -> MobileNetV2:
        """MobileNetV2 architecture from the `MobileNetV2: Inverted Residuals and Linear
        Bottlenecks <https://arxiv.org/abs/1801.04381>`_ paper.

        Args:
            weights (:class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
                pretrained weights to use. See
                :class:`~torchvision.models.MobileNet_V2_Weights` below for
                more details, and possible values. By default, no pre-trained
                weights are used.
            progress (bool, optional): If True, displays a progress bar of the
                download to stderr. Default is True.
            **kwargs: parameters passed to the ``torchvision.models.mobilenetv2.MobileNetV2``
                base class. Please refer to the `source code
                <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py>`_
                for more details about this class.

        .. autoclass:: torchvision.models.MobileNet_V2_Weights
            :members:
        """
        if kwargs['num_classes'] != 1000:
            num_classes = kwargs['num_classes']
            kwargs['num_classes'] = 1000
        weights = MobileNet_V2_Weights.verify(weights)

        if weights is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

        model = MobileNetV2(**kwargs)

        if weights is not None:
            model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        
        if num_classes != 1000:
            w = model.classifier[1].weight[:10]
            b = model.classifier[1].bias[:10]
            model.classifier[1] = mod.FreezableLinear(model.last_channel, num_classes)
            model.classifier[1].weight = nn.Parameter(w)
            model.classifier[1].bias = nn.Parameter(b)

        return model