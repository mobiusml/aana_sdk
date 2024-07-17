from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from skimage import transform
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)

from aana.configs.settings import settings

adaface_models = {
    "ir_50_msceleb": {
        "repo_id": "mobiuslabsgmbh/aana_facerecognition",
        "filename": "face_verification/adaface_ir50_ms1mv2.ckpt",
    },
    "ir_50_webface4M": {
        "repo_id": "mobiuslabsgmbh/aana_facerecognition",
        "filename": "face_verification/adaface_ir50_webface4m.ckpt",
    },
    "ir_101_webface4M": {
        "repo_id": "mobiuslabsgmbh/aana_facerecognition",
        "filename": "face_verification/adaface_ir101_webface4m.ckpt",
    },
    "ir_101_msceleb": {
        "repo_id": "mobiuslabsgmbh/aana_facerecognition",
        "filename": "face_verification/adaface_ir101_ms1mv3.ckpt",
    },
}


def face_align_landmarks(img, landmarks, image_size=(112, 112), method="similar"):
    tform = (
        transform.AffineTransform()
        if method == "affine"
        else transform.SimilarityTransform()
    )

    src = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.729904, 92.2041],
        ],
        dtype=np.float32,
    )
    ret = []
    landmarks = (
        landmarks if landmarks.shape[1] == 5 else np.reshape(landmarks, [-1, 5, 2])
    )
    for landmark in landmarks:
        tform.estimate(landmark, src)
        ndimage = transform.warp(img, tform.inverse, output_shape=image_size)
        if len(ndimage.shape) == 2:
            ndimage = np.stack([ndimage, ndimage, ndimage], -1)
        ret.append(ndimage)
    # return np.array(ret)
    return (np.array(ret) * 255).astype(np.uint8)


def to_input(np_img_list, device):
    """Convert a list of numpy images to a tensor suitable for model input.

    Args:
    np_img_list (list of np.array): List of images in numpy array format.

    Returns:
    torch.Tensor: Tensor containing all processed images.
    """
    # Convert list of images to a single numpy array (batch_size, height, width, channels)
    np_imgs = np.stack(np_img_list, axis=0)

    # Convert from RGB to BGR by reversing the last dimension
    np_imgs = np_imgs[:, :, :, ::-1]

    # Normalize the images to the range [-1, 1]
    np_imgs = (np_imgs / 255.0 - 0.5) / 0.5

    # Transpose the dimensions to (batch_size, channels, height, width)
    np_imgs = np_imgs.transpose(0, 3, 1, 2)

    # Convert to a torch tensor
    tensor = torch.tensor(np_imgs).float().contiguous().to(device)

    return tensor


def load_pretrained_model(architecture):
    # arch_name = '_'.join(architecture.split('_')[0:-2])
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = build_model(architecture)
    model_weights_path = hf_hub_download(
        repo_id=adaface_models[architecture]["repo_id"],
        filename=adaface_models[architecture]["filename"],
        local_dir=settings.model_dir,
    )
    statedict = torch.load(model_weights_path)["state_dict"]
    model_statedict = {
        key[6:]: val for key, val in statedict.items() if key.startswith("model.")
    }
    model.load_state_dict(model_statedict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device


# def load_IR50_model(path_to_weights):
#     # load model and pretrained statedict
#     # assert architecture in adaface_models.keys()
#     model = build_model("ir_50")
#     statedict = torch.load(path_to_weights)["state_dict"]
#     model_statedict = {
#         key[6:]: val for key, val in statedict.items() if key.startswith("model.")
#     }
#     model.load_state_dict(model_statedict)

#     # Move the model to GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.eval()
#     return model, device


# def load_pretrained_model(architecture="ir_50"):
#     # load model and pretrained statedict
#     # assert architecture in adaface_models.keys()
#     model = build_model(architecture)
#     statedict = torch.load(adaface_models[architecture])["state_dict"]
#     model_statedict = {
#         key[6:]: val for key, val in statedict.items() if key.startswith("model.")
#     }
#     model.load_state_dict(model_statedict)
#     model.eval()
#     return model


def build_model(model_name):
    if "ir_101" in model_name:
        return IR_101(input_size=(112, 112))
    elif "ir_50" in model_name:
        return IR_50(input_size=(112, 112))
    else:
        raise ValueError("not a correct model name", model_name)


def initialize_weights(modules):
    """Weight initilize, conv2d and linear is initialized with kaiming_normal"""
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()


class Flatten(Module):
    """Flat tensor"""

    def forward(self, input):
        return input.view(input.size(0), -1)


class LinearBlock(Module):
    """Convolution block without no-linear activation layer"""

    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(
            in_c, out_c, kernel, stride, padding, groups=groups, bias=False
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GNAP(Module):
    """Global Norm-Aware Pooling block"""

    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(in_c, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    """Global Depthwise Convolution block"""

    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = LinearBlock(
            in_c, in_c, groups=in_c, kernel=(7, 7), stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_flatten = Flatten()
        self.linear = Linear(in_c, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size, affine=False)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class SEModule(Module):
    """SE block"""

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class BasicBlockIR(Module):
    """BasicBlock for IRNet"""

    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BottleneckIR(Module):
    """BasicBlock with bottleneck for IRNet"""

    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for i in range(num_units - 1)
    ]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2),
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3),
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3),
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode="ir"):
        """Args:
        input_size: input_size of backbone
        num_layers: num_layers of backbone
        mode: support ir or irse
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [
            112,
            224,
        ], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [
            18,
            34,
            50,
            100,
            152,
            200,
        ], "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == "ir":
                unit_module = BasicBlockIR
            elif mode == "ir_se":
                unit_module = BasicBlockIRSE
            output_channel = 512
        else:
            if mode == "ir":
                unit_module = BottleneckIR
            elif mode == "ir_se":
                unit_module = BottleneckIRSE
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel),
                Dropout(0.4),
                Flatten(),
                Linear(output_channel * 7 * 7, 512),
                BatchNorm1d(512, affine=False),
            )
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel),
                Dropout(0.4),
                Flatten(),
                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False),
            )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        initialize_weights(self.modules())

    def forward(self, x):
        # current code only supports one extra image
        # it comes with a extra dimension for number of extra image. We will just squeeze it out for now
        x = self.input_layer(x)

        for idx, module in enumerate(self.body):
            x = module(x)

        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)

        return output, norm


def IR_18(input_size):
    """Constructs a ir-18 model."""
    model = Backbone(input_size, 18, "ir")

    return model


def IR_34(input_size):
    """Constructs a ir-34 model."""
    model = Backbone(input_size, 34, "ir")

    return model


def IR_50(input_size):
    """Constructs a ir-50 model."""
    model = Backbone(input_size, 50, "ir")

    return model


def IR_101(input_size):
    """Constructs a ir-101 model."""
    model = Backbone(input_size, 100, "ir")

    return model


def IR_152(input_size):
    """Constructs a ir-152 model."""
    model = Backbone(input_size, 152, "ir")

    return model


def IR_200(input_size):
    """Constructs a ir-200 model."""
    model = Backbone(input_size, 200, "ir")

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    model = Backbone(input_size, 50, "ir_se")

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model."""
    model = Backbone(input_size, 100, "ir_se")

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model."""
    model = Backbone(input_size, 152, "ir_se")

    return model


def IR_SE_200(input_size):
    """Constructs a ir_se-200 model."""
    model = Backbone(input_size, 200, "ir_se")

    return model
