import os
import unittest

import torch
from torch import nn

from model.mm_gtunets import CustomConv2d


class OriginalCustomConv2d(nn.Conv2d):
    def __init__(self):
        super().__init__(3, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        self.weight.data[:, 1, :, :] = -torch.max(
            torch.abs(self.weight.data[:, 1, :, :]),
            self.weight.data[:, 1, :, :] + self.weight.data[:, 2, :, :],
        )
        self.weight.data[:, 0, :, :] = torch.max(
            self.weight.data[:, 0, :, :], self.weight.data[:, 2, :, :]
        )
        return super().forward(x)


class PPUCustomConvEquivalenceTest(unittest.TestCase):
    def test_forward_and_gradients_match_original_conv2d(self):
        if not torch.cuda.is_available():
            self.fail("PPU/CUDA device required; CPU validation is not accepted")

        device = torch.device(os.environ.get("MM_PPU_TEST_DEVICE", "cuda:0"))
        device_name = torch.cuda.get_device_name(device)
        self.assertIn("PPU", device_name.upper(), device_name)
        print(f"device={device} ({device_name}) torch={torch.__version__}")

        torch.manual_seed(911)
        initial_weight = torch.tensor([[-0.7, -0.3, 0.2]], device=device).view(1, 3, 1, 1)
        cases = [
            ("small", (3, 19, 17)),
            ("abide_graph", (3, 871, 871)),
        ]

        for name, shape in cases:
            with self.subTest(case=name):
                original = OriginalCustomConv2d().to(device)
                compatible = CustomConv2d(3, 1).to(device)
                with torch.no_grad():
                    original.weight.copy_(initial_weight)
                    compatible.weight.copy_(initial_weight)

                base_x = torch.randn(shape, device=device)
                x_original = base_x.detach().clone().requires_grad_(True)
                x_compatible = base_x.detach().clone().requires_grad_(True)

                y_original = original(x_original)
                y_compatible = compatible(x_compatible)
                probe = torch.randn_like(y_original)
                grad_x_original, grad_w_original = torch.autograd.grad(
                    (y_original * probe).sum(), (x_original, original.weight)
                )
                grad_x_compatible, grad_w_compatible = torch.autograd.grad(
                    (y_compatible * probe).sum(), (x_compatible, compatible.weight)
                )
                torch.cuda.synchronize(device)

                torch.testing.assert_close(y_compatible, y_original, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(grad_x_compatible, grad_x_original, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(grad_w_compatible, grad_w_original, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(compatible.weight, original.weight, rtol=0, atol=0)

                print(
                    f"{name}: output_max_abs="
                    f"{(y_compatible - y_original).abs().max().item():.3e}, "
                    f"input_grad_max_abs="
                    f"{(grad_x_compatible - grad_x_original).abs().max().item():.3e}, "
                    f"weight_grad_max_abs="
                    f"{(grad_w_compatible - grad_w_original).abs().max().item():.3e}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
