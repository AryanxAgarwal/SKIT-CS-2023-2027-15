import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union

class ConvolutionalShapeValidator:
    """
    Validates and traces input/output tensor shapes across convolutional 
    feature extraction networks using both analytical formulas and runtime hooks.
    """
    def __init__(self, model: nn.Module, input_size: Tuple[int, ...]):
        self.model = model
        self.input_size = input_size
        self.layer_records: List[Dict[str, Union[str, List[int]]]] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _calc_conv_output_shape(
        self, 
        in_dim: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]] = 1, 
        padding: Union[int, Tuple[int, int], str] = 0, 
        dilation: Union[int, Tuple[int, int]] = 1
    ) -> int:
        """Calculates expected output spatial dimension for convolutions/pooling."""
        if isinstance(kernel_size, int):
            ks_h, ks_w = kernel_size, kernel_size
        else:
            ks_h, ks_w = kernel_size

        if isinstance(stride, int):
            stride_h, stride_w = stride, stride
        else:
            stride_h, stride_w = stride

        if isinstance(padding, int):
            pad_h, pad_w = padding, padding
        elif isinstance(padding, tuple):
            pad_h, pad_w = padding
        else:
            # Handle 'same' or 'valid' if necessary, default to 0 for simplicity
            pad_h, pad_w = 0, 0

        if isinstance(dilation, int):
            dil_h, dil_w = dilation, dilation
        else:
            dil_h, dil_w = dilation

        out_h = (in_dim[0] + 2 * pad_h - dil_h * (ks_h - 1) - 1) // stride_h + 1
        out_w = (in_dim[1] + 2 * pad_w - dil_w * (ks_w - 1) - 1) // stride_w + 1
        return out_h, out_w

    def _get_forward_hook(self, layer_name: str):
        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            in_tensor = input[0]
            record = {
                "layer_name": layer_name,
                "layer_type": module.__class__.__name__,
                "input_shape": list(in_tensor.shape),
                "output_shape": list(output.shape),
            }
            self.layer_records.append(record)
        return hook

    def register_hooks(self) -> None:
        """Registers forward hooks on all convolutional and pooling layers."""
        target_classes = (
            nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
            nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
            nn.BatchNorm2d, nn.ReLU, nn.AdaptiveAvgPool2d
        )
        for name, module in self.model.named_modules():
            if isinstance(module, target_classes):
                handle = module.register_forward_hook(self._get_forward_hook(name))
                self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """Cleans up registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def validate(self) -> List[Dict[str, Union[str, List[int]]]]:
        """
        Runs a dummy forward pass through the model to capture shapes 
        and validate tensor flow continuity.
        """
        self.layer_records.clear()
        self.register_hooks()

        device = next(self.model.parameters()).device
        dummy_input = torch.randn(*self.input_size, device=device)

        try:
            self.model.eval()
            with torch.no_grad():
                _ = self.model(dummy_input)
        except Exception as e:
            raise RuntimeError(f"Shape validation failed during forward pass: {e}")
        finally:
            self.remove_hooks()

        return self.layer_records

    def print_summary(self) -> None:
        """Prints a clean tabular report of input/output tensor shapes per layer."""
        records = self.validate()
        
        print("=" * 95)
        print(f"{'Layer Name':<30} | {'Layer Type':<18} | {'Input Shape':<20} | {'Output Shape':<20}")
        print("-" * 95)
        
        for rec in records:
            name = rec["layer_name"] if rec["layer_name"] else "root"
            l_type = rec["layer_type"]
            in_shape = str(rec["input_shape"])
            out_shape = str(rec["output_shape"])
            print(f"{name:<30} | {l_type:<18} | {in_shape:<20} | {out_shape:<20}")
        
        print("=" * 95)


# --- Example Usage ---
if __name__ == "__main__":
    class SampleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Linear(32 * 8 * 8, 10)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # Initialize model and validator for an input tensor of batch size 4, 3 channels, 32x32 image
    model = SampleCNN()
    validator = ConvolutionalShapeValidator(model, input_size=(4, 3, 32, 32))
    
    # Run validation and print layer shape flow
    validator.print_summary()