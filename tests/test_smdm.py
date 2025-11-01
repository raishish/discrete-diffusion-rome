"""Tests for testing inference for SMDM models."""
import sys
from pathlib import Path

import torch
import pytest
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# Add parent directory to path to import smdm
sys.path.insert(0, str(Path(__file__).parent.parent))

from smdm import get_config_class, ARModel, MaskedDiffusionModel


def get_available_devices():
    """Get list of available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture
def config():
    """Load the Diff_LLaMA_19M config."""
    return get_config_class("Diff_LLaMA_19M")


@pytest.fixture(params=get_available_devices())
def device(request):
    """Parametrized fixture for available devices."""
    return request.param


@pytest.fixture
def dummy_input(device):
    """Create a dummy input tensor."""
    return torch.randint(0, 32000, (1, 52)).to(device)


@pytest.fixture
def ar_checkpoint_path():
    """Download AR checkpoint from HuggingFace Hub and return path."""
    return hf_hub_download(
        repo_id="nieshen/SMDM",
        filename="ar_safetensors/ar-19M-10e18.safetensors",
        repo_type="model"
    )


@pytest.fixture
def mdm_checkpoint_path():
    """Download MDM checkpoint from HuggingFace Hub and return path."""
    return hf_hub_download(
        repo_id="nieshen/SMDM",
        filename="mdm_safetensors/mdm-19M-10e18.safetensors",
        repo_type="model"
    )


class TestARModelInference:
    """Tests for the Autoregressive Model."""

    def test_armodel_forward_pass(self, config, device, dummy_input, ar_checkpoint_path):
        """Test ARModel: create, move to device, forward pass, and load pretrained weights."""
        # Step 1: Create model
        model = ARModel(config)
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0, "Model should have parameters"

        # Step 2: Move to device
        model = model.to(device)
        for param in model.parameters():
            assert param.device.type == (device if device == "cpu" else device.split(":")[0])

        # Step 3: Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 52, config.padded_vocab_size), \
            f"[{device}] Forward pass shape mismatch: {output.shape}"

        # Step 4: Load pretrained weights
        state_dict = load_file(ar_checkpoint_path)
        model.load_state_dict(state_dict)

        # Step 5: Inference with pretrained weights
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 52, config.padded_vocab_size), \
            f"[{device}] Pretrained inference shape mismatch: {output.shape}"


class TestMaskedDiffusionModelInference:
    """Tests for the Masked Diffusion Model."""

    def test_mdm_forward_pass(self, config, device, dummy_input, mdm_checkpoint_path):
        """Test MaskedDiffusionModel: create, move to device, forward pass, and load pretrained weights."""
        # Step 1: Create model
        model = MaskedDiffusionModel(config)
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0, "Model should have parameters"

        # Step 2: Move to device
        model = model.to(device)
        for param in model.parameters():
            assert param.device.type == (device if device == "cpu" else device.split(":")[0])

        # Step 3: Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 52, config.padded_vocab_size), \
            f"[{device}] Forward pass shape mismatch: {output.shape}"

        # Step 4: Load pretrained weights
        state_dict = load_file(mdm_checkpoint_path)
        model.load_state_dict(state_dict)

        # Step 5: Inference with pretrained weights
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 52, config.padded_vocab_size), \
            f"[{device}] Pretrained inference shape mismatch: {output.shape}"
