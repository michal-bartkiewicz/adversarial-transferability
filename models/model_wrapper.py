import torch.nn as nn

class ModelWrapper(nn.Module):
    """
    A wrapper for PyTorch models that adds a .name property and handles metadata.
    """
    def __init__(self, model: nn.Module, arch_name: str, pretrained: bool):
        """
        Initializes the ModelWrapper.

        Args:
            model (nn.Module): The underlying PyTorch model.
            arch_name (str): The name of the architecture (e.g., 'resnet18').
            pretrained (bool): Whether the model was initialized with pretrained weights.
        """
        super(ModelWrapper, self).__init__()
        self.model = model
        self.arch_name = arch_name
        self.pretrained = pretrained
        self._name = f"{arch_name}_{'pretrained' if pretrained else 'scratch'}"

    @property
    def name(self) -> str:
        """
        Returns the standardized name of the model.

        Returns:
            str: Model name in format {architecture}_{weights_status}.
        """
        return self._name

    def forward(self, x):
        """
        Forward pass through the underlying model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        return self.model(x)
