"""
Sprint 2 - Unit Testing & Feature Layer Validation
Goal: validate tensor input/output shapes across convolutional feature extraction layers.

Run:
    python test_feature_layer_shapes.py          (unittest)
    pytest -v test_feature_layer_shapes.py       (pytest also works)

HOW TO USE WITH YOUR PROJECT
    1. Replace `ConvFeatureExtractor` (or just edit `build_model()`) so it returns YOUR model.
    2. Your model must expose its conv stack as `model.features` (an nn.Sequential).
       If it's named differently, change FEATURES_ATTR below.
    3. Adjust INPUT_SHAPE / EXPECTED_OUT_DIM to match your data.
"""
import unittest
from contextlib import contextmanager

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
FEATURES_ATTR = "features"          # attribute holding the conv stack
INPUT_SHAPE = (3, 64, 64)           # (C, H, W) of ONE sample
BATCH_SIZES = (1, 4, 8)
EXPECTED_OUT_DIM = 64               # final feature-vector length (last conv channels)


# --------------------------------------------------------------------------- #
# Example model (stand-in) - replace with your own
# --------------------------------------------------------------------------- #
class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, channels=(16, 32, 64), kernel_size=3, padding=1):
        super().__init__()
        layers, prev = [], in_channels
        for c in channels:
            layers += [
                nn.Conv2d(prev, c, kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = c
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return torch.flatten(self.gap(self.features(x)), 1)


def build_model():
    return ConvFeatureExtractor(in_channels=INPUT_SHAPE[0])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def _out_size(size, kernel, stride, padding, dilation):
    """Standard conv/pool output size formula (floor mode)."""
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def expected_layer_shapes(features, input_shape):
    """
    Analytically compute the expected output shape of every leaf layer
    (independent of running the model), starting from input_shape = (N, C, H, W).
    Returns a list of (layer_name, expected_output_shape).
    """
    n, c, h, w = input_shape
    out = []
    for name, m in features.named_modules():
        if len(list(m.children())) > 0:   # skip containers
            continue
        if isinstance(m, nn.Conv2d):
            kh, kw = _pair(m.kernel_size)
            sh, sw = _pair(m.stride)
            ph, pw = _pair(m.padding)
            dh, dw = _pair(m.dilation)
            h = _out_size(h, kh, sh, ph, dh)
            w = _out_size(w, kw, sw, pw, dw)
            c = m.out_channels
        elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            kh, kw = _pair(m.kernel_size)
            sh, sw = _pair(m.stride if m.stride is not None else m.kernel_size)
            ph, pw = _pair(m.padding)
            dh, dw = _pair(getattr(m, "dilation", 1))
            h = _out_size(h, kh, sh, ph, dh)
            w = _out_size(w, kw, sw, pw, dw)
        # BatchNorm / ReLU / Dropout / etc. keep the shape unchanged
        out.append((name, (n, c, h, w)))
    return out


@contextmanager
def record_shapes(features):
    """Forward-hook every leaf layer and record (name, module, in_shape, out_shape)."""
    records, handles = [], []

    def make_hook(name):
        def hook(module, inputs, output):
            records.append((name, module, tuple(inputs[0].shape), tuple(output.shape)))
        return hook

    for name, m in features.named_modules():
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(make_hook(name)))
    try:
        yield records
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestFeatureLayerShapes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_model().eval()
        self.features = getattr(self.model, FEATURES_ATTR)

    # -- 1. every layer's actual output matches the analytically expected shape
    def test_each_layer_output_shape_matches_expected(self):
        for bs in BATCH_SIZES:
            with self.subTest(batch_size=bs):
                x = torch.randn(bs, *INPUT_SHAPE)
                expected = expected_layer_shapes(self.features, tuple(x.shape))
                with record_shapes(self.features) as rec, torch.no_grad():
                    self.model(x)
                self.assertEqual(len(rec), len(expected), "Layer count mismatch")
                for (name, _, _, got), (_, exp) in zip(rec, expected):
                    self.assertEqual(got, exp, f"Layer '{name}': got {got}, expected {exp}")

    # -- 2. output of layer i is the input of layer i+1 (no silent reshapes)
    def test_shape_continuity_between_layers(self):
        x = torch.randn(2, *INPUT_SHAPE)
        with record_shapes(self.features) as rec, torch.no_grad():
            self.model(x)
        for (n1, _, _, out1), (n2, _, in2, _) in zip(rec, rec[1:]):
            self.assertEqual(out1, in2, f"Break between '{n1}' and '{n2}': {out1} -> {in2}")

    # -- 3. conv in_channels matches incoming tensor channels; out_channels matches output
    def test_conv_channel_consistency(self):
        x = torch.randn(2, *INPUT_SHAPE)
        with record_shapes(self.features) as rec, torch.no_grad():
            self.model(x)
        convs = [r for r in rec if isinstance(r[1], nn.Conv2d)]
        self.assertGreater(len(convs), 0, "No Conv2d layers found")
        for name, m, in_shape, out_shape in convs:
            self.assertEqual(in_shape[1], m.in_channels, f"{name}: input channels")
            self.assertEqual(out_shape[1], m.out_channels, f"{name}: output channels")

    # -- 4. batch dimension is preserved through all layers
    def test_batch_dimension_preserved(self):
        for bs in BATCH_SIZES:
            with self.subTest(batch_size=bs):
                x = torch.randn(bs, *INPUT_SHAPE)
                with record_shapes(self.features) as rec, torch.no_grad():
                    out = self.model(x)
                self.assertTrue(all(r[3][0] == bs for r in rec))
                self.assertEqual(out.shape[0], bs)

    # -- 5. final feature vector shape
    def test_final_feature_vector_shape(self):
        for bs in BATCH_SIZES:
            with self.subTest(batch_size=bs):
                out = self.model(torch.randn(bs, *INPUT_SHAPE))
                self.assertEqual(tuple(out.shape), (bs, EXPECTED_OUT_DIM))

    # -- 6. different spatial sizes still produce the same-length feature vector
    def test_variable_spatial_size(self):
        c = INPUT_SHAPE[0]
        for hw in [(32, 32), (64, 64), (96, 128)]:
            with self.subTest(hw=hw):
                out = self.model(torch.randn(2, c, *hw))
                self.assertEqual(tuple(out.shape), (2, EXPECTED_OUT_DIM))

    # -- 7. wrong channel count / wrong rank must raise
    def test_invalid_inputs_raise(self):
        bad_channels = torch.randn(2, INPUT_SHAPE[0] + 1, *INPUT_SHAPE[1:])
        with self.assertRaises(RuntimeError):
            self.model(bad_channels)
        with self.assertRaises((RuntimeError, ValueError, IndexError)):
            self.model(torch.randn(*INPUT_SHAPE))  # missing batch dim (3D)

    # -- 8. outputs are finite and dtype is preserved
    def test_outputs_finite_and_dtype(self):
        out = self.model(torch.randn(4, *INPUT_SHAPE))
        self.assertTrue(torch.isfinite(out).all().item(), "NaN/Inf in features")
        self.assertEqual(out.dtype, torch.float32)

    # -- 9. train vs eval mode give the same output SHAPES
    def test_train_eval_shape_parity(self):
        x = torch.randn(4, *INPUT_SHAPE)
        self.model.eval()
        with torch.no_grad():
            eval_out = self.model(x)
        self.model.train()
        train_out = self.model(x)
        self.assertEqual(eval_out.shape, train_out.shape)

    # -- 10. gradients flow back with the same shape as the input
    def test_backward_gradient_shapes(self):
        self.model.train()
        x = torch.randn(4, *INPUT_SHAPE, requires_grad=True)
        self.model(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, f"No grad for {name}")
            self.assertEqual(p.grad.shape, p.shape, f"Grad shape mismatch: {name}")

    # -- 11. batch-size invariance: sample alone vs inside a batch (eval mode)
    def test_batch_invariance(self):
        self.model.eval()
        x = torch.randn(4, *INPUT_SHAPE)
        with torch.no_grad():
            batched = self.model(x)
            single = self.model(x[:1])
        self.assertTrue(torch.allclose(batched[:1], single, atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
