# models/bisenet/build_bisenet_multilevel.py

import torch
from torch import nn

# --- NEW: Import shared components from the original BiSeNet file ---
from .build_bisenet import AttentionRefinementModule, FeatureFusionModule, Spatial_path
from .build_contextpath import build_contextpath


class BiSeNetMultiLevel(torch.nn.Module):
    """
    This is the multi-level version of BiSeNet. It inherits all the component
    classes from the original build_bisenet.py file and only redefines the
    main class structure and its forward pass to return an intermediate feature map.
    """

    def __init__(self, num_classes, context_path):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path)

        # build attention refinement module for resnet 101
        if context_path == "resnet101":
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            # supervision block
            self.supervision1 = nn.Conv2d(
                in_channels=1024, out_channels=num_classes, kernel_size=1
            )
            self.supervision2 = nn.Conv2d(
                in_channels=2048, out_channels=num_classes, kernel_size=1
            )
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 3328)

        elif context_path == "resnet18":
            # build attention refinement module for resnet 18
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            # supervision block
            self.supervision1 = nn.Conv2d(
                in_channels=256, out_channels=num_classes, kernel_size=1
            )
            self.supervision2 = nn.Conv2d(
                in_channels=512, out_channels=num_classes, kernel_size=1
            )
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)
        else:
            raise ValueError(f"Unsupported context_path network: {context_path}")

        # build final convolution
        self.conv = nn.Conv2d(
            in_channels=num_classes, out_channels=num_classes, kernel_size=1
        )

        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.saptial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.supervision1)
        self.mul_lr.append(self.supervision2)
        self.mul_lr.append(self.feature_fusion_module)
        self.mul_lr.append(self.conv)

    def init_weight(self):
        for name, m in self.named_modules():
            if "context_path" not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)

        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, size=sx.size()[-2:], mode="bilinear")
        cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode="bilinear")
        cx = torch.cat((cx1, cx2), dim=1)

        if self.training:
            # Intermediate supervision branches from the original BiSeNet paper
            cx1_sup = self.supervision1(cx1)
            cx2_sup = self.supervision2(cx2)
            cx1_sup = torch.nn.functional.interpolate(
                cx1_sup, size=input.size()[-2:], mode="bilinear"
            )
            cx2_sup = torch.nn.functional.interpolate(
                cx2_sup, size=input.size()[-2:], mode="bilinear"
            )

        # output of feature fusion module
        fused_features = self.feature_fusion_module(sx, cx)

        # upsampling to final output size
        final_result = torch.nn.functional.interpolate(
            fused_features, scale_factor=8, mode="bilinear"
        )
        final_result = self.conv(final_result)

        if self.training:
            # Return the final result, the two standard supervision outputs,
            # and the intermediate fused_features for our auxiliary discriminator.
            return final_result, cx1_sup, cx2_sup, fused_features

        return final_result
