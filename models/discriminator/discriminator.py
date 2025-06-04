import torch
import torch.nn as nn


class FCDiscriminator(nn.Module):
    """
    Fully Convolutional Discriminator network based on Tsai et al. (2017) / AdvEnt.
    Takes a probability map (e.g., from a segmentation network) as input and
    outputs a map of scores indicating whether the input is from the source or target domain.
    """

    def __init__(self, num_classes: int, ndf: int = 64):
        """
        Initializes the discriminator.

        Args:
            num_classes (int): Number of input channels (typically the number of segmentation classes).
            ndf (int): Number of discriminator filters in the first convolutional layer.
                       Subsequent layers will have multiples of this.
        """
        super(FCDiscriminator, self).__init__()

        # Layer 1: Input (Batch size, num_classes, H, W) -> Output (Batch size, ndf, H/2, W/2)
        # Paper: kernel 4x4, stride 2. Channels: {64, 128, 256, 512, 1}
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Layer 2: Input (B, ndf, H/2, W/2) -> Output (B, ndf*2, H/4, W/4)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Layer 3: Input (B, ndf*2, H/4, W/4) -> Output (B, ndf*4, H/8, W/8)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Layer 4: Input (B, ndf*4, H/8, W/8) -> Output (B, ndf*8, H/16, W/16)
        # Paper mentions 512 channels for the fourth conv layer if ndf=64, so ndf*8 = 64*8 = 512.
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Layer 5: Input (B, ndf*8, H/16, W/16) -> Output (B, 1, H_out, W_out)
        # The paper states the last layer channel is 1.
        # The paper also mentions "An up-sampling layer is added to the last convolution layer for
        # re-scaling the output to the size of the input."
        # For a fully convolutional discriminator outputting a spatial map of logits for adversarial loss,
        # direct upsampling to input size is not always done if loss is applied spatially.
        # We will output a smaller map and apply loss spatially.
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)

        # Weight initialization (optional, but can be helpful)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights of the convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Normal initialization as often used in GANs (e.g., DCGAN paper)
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input tensor, expected to be probability maps from the
                              segmentation network. Shape: (B, num_classes, H, W).

        Returns:
            torch.Tensor: Output logits map. Shape: (B, 1, H_out, W_out).
                          These logits are then used with BCEWithLogitsLoss.
                          H_out and W_out will be H/32 and W/32 of the input.
        """
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.leaky_relu4(x)
        x = self.classifier(x)  # Final layer to produce logits
        return x


if __name__ == "__main__":
    # Example usage:
    num_classes_example = 19  # e.g., Cityscapes
    batch_size_example = 2

    # Test with Cityscapes-like dimensions
    height_cs, width_cs = 512, 1024
    dummy_input_cs = torch.rand(
        batch_size_example, num_classes_example, height_cs, width_cs
    )
    discriminator_cs = FCDiscriminator(num_classes=num_classes_example)
    output_logits_map_cs = discriminator_cs(dummy_input_cs)
    print("Discriminator instantiated for Cityscapes-like input.")
    print(
        f"Number of parameters in Discriminator: {sum(p.numel() for p in discriminator_cs.parameters() if p.requires_grad) / 1e6:.2f} M"
    )
    print(f"Input shape (Cityscapes-like): {dummy_input_cs.shape}")
    # Expected output H_out = H_in / (2^5) = 512 / 32 = 16
    # Expected output W_out = W_in / (2^5) = 1024 / 32 = 32
    print(
        f"Output logits map shape (Cityscapes-like): {output_logits_map_cs.shape}"
    )  # Expected: (B, 1, 16, 32)

    # Test with GTA5-like dimensions
    height_gta, width_gta = 720, 1280
    dummy_input_gta = torch.rand(
        batch_size_example, num_classes_example, height_gta, width_gta
    )
    discriminator_gta = FCDiscriminator(
        num_classes=num_classes_example
    )  # Can reuse or make new
    output_logits_map_gta = discriminator_gta(dummy_input_gta)
    print("\nDiscriminator instantiated for GTA5-like input.")
    print(f"Input shape (GTA5-like): {dummy_input_gta.shape}")
    # Expected output H_out = 720 / 32 = 22.5 -> floor = 22
    # Expected output W_out = 1280 / 32 = 40
    print(
        f"Output logits map shape (GTA5-like): {output_logits_map_gta.shape}"
    )  # Expected: (B, 1, 22, 40)
