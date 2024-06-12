import torch
import torch.nn.functional as F
from torch import nn
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchEmbedding(nn.Module):
    """
    Path embedding layer is nothing but a convolutional layer with kerneli size and stride equal to patch size.
    """

    def __init__(self, in_channels, embedding_dim, patch_size):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels, embedding_dim, patch_size, patch_size
        )

    def forward(self, x):
        return self.patch_embedding(x)


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Calculate the grid step size
        grid_step = 2 / grid_size

        # Create the grid tensor
        grid_range = torch.arange(-spline_order, grid_size + spline_order + 1)
        grid_values = grid_range * grid_step - 1
        self.grid = grid_values.expand(in_features, -1).contiguous()

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.base_activation = nn.SiLU()

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the base weight tensor with Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        with torch.no_grad():
            # Generate random noise for initializing the spline weights
            noise_shape = (self.grid_size + 1, self.in_features, self.out_features)
            random_noise = (torch.rand(noise_shape) - 0.5) * 0.1 / self.grid_size

            # Compute the spline weight coefficients from the random noise
            grid_points = self.grid.T[self.spline_order : -self.spline_order]
            spline_coefficients = self.curve2coeff(grid_points, random_noise)

            # Copy the computed coefficients to the spline weight tensor
            self.spline_weight.data.copy_(spline_coefficients)

        # Initialize the spline scaler tensor with Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """

        # Expand the grid tensor to match the input tensor's dimensions
        expanded_grid = (
            self.grid.unsqueeze(0).expand(x.size(0), *self.grid.size()).to(device)
        )  # (batch_size, in_features, grid_size + 2 * spline_order + 1)

        # Add an extra dimension to the input tensor for broadcasting
        input_tensor_expanded = x.unsqueeze(-1).to(
            device
        )  # (batch_size, in_features, 1)

        # Initialize the bases tensor with boolean values
        bases = (
            (input_tensor_expanded >= expanded_grid[:, :, :-1])
            & (input_tensor_expanded < expanded_grid[:, :, 1:])
        ).to(x.dtype)  # (batch_size, in_features, grid_size + spline_order)

        # Compute the B-spline bases recursively
        for order in range(1, self.spline_order + 1):
            left_term = (
                (input_tensor_expanded - expanded_grid[:, :, : -order - 1])
                / (expanded_grid[:, :, order:-1] - expanded_grid[:, :, : -order - 1])
            ) * bases[:, :, :-1]

            right_term = (
                (expanded_grid[:, :, order + 1 :] - input_tensor_expanded)
                / (expanded_grid[:, :, order + 1 :] - expanded_grid[:, :, 1:-order])
            ) * bases[:, :, 1:]

            bases = left_term + right_term

        return bases.contiguous()

    def curve2coeff(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        # Compute the B-spline bases for the input tensor
        b_splines_bases = self.b_splines(
            input_tensor
        )  # (batch_size, input_dim, grid_size + spline_order)

        # Transpose the B-spline bases and output tensor for matrix multiplication
        transposed_bases = b_splines_bases.transpose(
            0, 1
        )  # (input_dim, batch_size, grid_size + spline_order)
        transposed_output = output_tensor.transpose(
            0, 1
        )  # (input_dim, batch_size, output_dim)

        # Convert tensor into the current device type
        transposed_bases = transposed_bases.to(device)
        transposed_output = transposed_output.to(device)

        # Solve the least-squares problem to find the coefficients
        coefficients_solution = torch.linalg.lstsq(
            transposed_bases, transposed_output
        ).solution
        # (input_dim, grid_size + spline_order, output_dim)

        # Permute the coefficients to match the expected shape
        coefficients = coefficients_solution.permute(
            2, 0, 1
        )  # (output_dim, input_dim, grid_size + spline_order)

        return coefficients.contiguous()

    def forward(self, x: torch.Tensor):
        # Save the original shape
        original_shape = x.shape

        # Flatten the last two dimensions of the input
        x = x.contiguous().view(-1, self.in_features)

        base_output = F.linear(
            self.base_activation(x).to(device), self.base_weight.to(device)
        )

        # val_b_spline = self.b_splines(x).view(x.size(0), -1).to(device)
        # val_spline_weight = self.spline_weight.view(self.out_features, -1).to(device)

        # val_b_spline = val_b_spline.to(device)
        # val_spline_weight = val_spline_weight.to(device)

        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1).to(device),
            self.spline_weight.view(self.out_features, -1).to(device),
            # self.b_splines(x).view(x.size(0), -1),
            # self.spline_weight.view(self.out_features, -1),
            # val_b_spline,
            # val_spline_weight,
        )

        # Apply the linear transformation
        output = base_output + spline_output

        # Reshape the output to have the same shape as the input
        output = output.view(*original_shape[:-1], -1)

        return output


# Kolmogorov-Arnold Networks
class KAN(nn.Module):
    """
    This network applies 2 consecutive fully connected layers and is used in Token Mixer and Channel Mixer modules.
    """

    def __init__(self, dim, intermediate_dim, dropout=0.0, grid_size=5, spline_order=3):
        super().__init__()
        self.kan = nn.Sequential(
            KANLinear(dim, intermediate_dim, grid_size, spline_order),
            KANLinear(intermediate_dim, dim, grid_size, spline_order),
        )

    def forward(self, x):
        return self.kan(x)


class Transformation1(nn.Module):
    """
    The transformation that is used in Mixer Layer (the T) which just switches the 2nd and the 3rd dimensions and is applied before and after Token Mixing KANs
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (0, 2, 1))


class Transformation2(nn.Module):
    """
    The transformation that is applied right after the patch embedding layer and convert it's shape from (batch_size,  embedding_dim, sqrt(num_patches), sqrt(num_patches)) to (batch_size, num_patches, embedding_dim)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (0, 2, 3, 1)).reshape(x.shape[0], -1, x.shape[1])


class MixerLayer(nn.Module):
    """
    Mixer layer which consists of Token Mixer and Channel Mixer modules in addition to skip connections.
    intermediate_output = Token Mixer(input) + input
    final_output = Channel Mixer(intermediate_output) + intermediate_output
    """

    def __init__(
        self,
        embedding_dim,
        num_patch,
        token_intermediate_dim,
        channel_intermediate_dim,
        dropout=0.0,
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            Transformation1(),
            KAN(num_patch, token_intermediate_dim, dropout, grid_size, spline_order),
            Transformation1(),
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            KAN(
                embedding_dim,
                channel_intermediate_dim,
                dropout,
                grid_size,
                spline_order,
            ),
        )

    def forward(self, x):
        val_token_mixer = self.token_mixer(x).to(device)
        val_channel_mixer = self.channel_mixer(x).to(device)
        x = x.to(device)

        # x = x + self.token_mixer(x)  # Token mixer and skip connection
        # x = x + self.channel_mixer(x)  # Channel mixer and skip connection

        x = x + val_token_mixer  # Token mixer and skip connection
        x = x + val_channel_mixer  # Channel mixer and skip connection

        return x


class KANMixer(nn.Module):
    """
    KAN-Mixer Architecture:
    1-Applies 'Patch Embedding' at first.
    2-Applies 'Mixer Layer' N times in a row.
    3-Performs 'Global Average Pooling'
    4-The Learnt features are then passed to the classifier
    """

    def __init__(
        self,
        in_channels,
        embedding_dim,
        num_classes,
        patch_size,
        image_size,
        depth,
        token_intermediate_dim,
        channel_intermediate_dim,
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()

        self.num_patch = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Sequential(
            PatchEmbedding(in_channels, embedding_dim, patch_size),
            Transformation2(),
        )

        self.mixers = nn.ModuleList(
            [
                MixerLayer(
                    embedding_dim,
                    self.num_patch,
                    token_intermediate_dim,
                    channel_intermediate_dim,
                    grid_size,
                    spline_order,
                )
                for _ in range(depth)
            ]
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(nn.Linear(embedding_dim, num_classes))

    def forward(self, x):
        x = self.patch_embedding(x)  # Patch Embedding layer
        for mixer in self.mixers:  # Applying Mixer Layer N times
            x = mixer(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # Global Average Pooling

        return self.classifier(x)
