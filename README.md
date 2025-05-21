# CoreAudioML

![Module Tests](https://github.com/Alec-Wright/CoreAudioML/workflows/Python%20Package%20using%20Conda/badge.svg?branch=main)

### Machine learning for audio effects processing

This repo contains functions and classes used in training neural network models of audio effects such as guitar amplifiers, distortion effects, phasers and flangers. This repo is intended to be used as a submodule for the other repos that actually contain the trained models and training scripts, for example [amplifiers/distortion](https://github.com/Alec-Wright/NeuralGuitarAmpModelling) and [phaser/flanger](https://github.com/Alec-Wright/NeuralTimeVaryFX)

## Device Support (CPU, NVIDIA CUDA, Intel XPU)

CoreAudioML now supports computations on different hardware devices:
- **CPU:** Default, widely available.
- **NVIDIA GPUs:** Via CUDA.
- **Intel GPUs:** Via Intel® Extension for PyTorch (IPEX) and oneAPI.

Most relevant classes and functions in `networks.py` and `training.py` now accept a `device` string parameter (e.g., `'cpu'`, `'cuda:0'`, `'xpu:0'`) to control where computations are performed.

### Using with Intel GPUs (XPU)

To leverage Intel GPUs for training or inference with CoreAudioML, you need to set up your environment correctly.

**1. Prerequisites:**
   - An Intel GPU compatible with Intel® oneAPI (e.g., Intel integrated graphics Gen9+, Intel® Arc™ A-Series GPUs, Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series).
   - **Intel GPU Drivers:** Ensure you have the latest drivers installed for your operating system. These typically include support for OpenCL and Intel Level Zero.
   - **Intel® oneAPI Base Toolkit:** While IPEX can be installed via pip, some components from the oneAPI Base Toolkit might be beneficial or required for optimal performance or if building from source, particularly:
     - Intel® DPC++/C++ Compiler
     - Intel® oneDNN (deep neural network library)
     - Relevant runtime libraries.
     Refer to the [Intel® oneAPI installation guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) for detailed instructions. For many users, installing IPEX via pip (see below) will bring necessary runtime components.

**2. Installation of PyTorch and IPEX:**
   Install PyTorch and Intel® Extension for PyTorch (IPEX) using pip. It's recommended to use the versions provided by Intel for XPU support:
   ```bash
   # Follow instructions from https://pytorch.org/get-started/locally/ for PyTorch if specific versions are needed.
   # For Intel GPU support, Intel provides specific wheels:
   pip install torch torchvision torchaudio --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
   pip install intel-extension-for-pytorch
   ```
   Always check the [official IPEX GitHub page](https://github.com/intel/intel-extension-for-pytorch) for the latest installation instructions and compatible PyTorch versions.

**3. Specifying the Device in CoreAudioML:**
   Once your environment is set up and IPEX is installed, you can specify an Intel GPU device to CoreAudioML components:
   - The `device` parameter in relevant classes and functions should be set to `'xpu'` or `'xpu:n'` (where `n` is the GPU index, e.g., `'xpu:0'`).

   **Example:**
   ```python
   import torch
   from CoreAudioML.networks import SimpleRNN # Assuming SimpleRNN is a class in CoreAudioML
   # from CoreAudioML.training import ESRLoss # Example loss function

   if __name__ == '__main__':
       target_device_str = 'cpu' # Default
       if hasattr(torch, 'xpu') and torch.xpu.is_available():
           print("Intel XPU is available. Using 'xpu:0'.")
           target_device_str = 'xpu:0'
       elif hasattr(torch, 'cuda') and torch.cuda.is_available():
           print("NVIDIA CUDA is available. Using 'cuda:0'.")
           target_device_str = 'cuda:0'
       else:
           print("No specialized GPU detected. Using 'cpu'.")

       # When creating a model:
       # model = SimpleRNN(input_size=1, output_size=1, hidden_size=32, device=target_device_str)
       # model.to(target_device_str) # Ensure model is on device, though constructor should handle it.

       # When creating a loss function:
       # loss_fn = ESRLoss(device=target_device_str)
       # loss_fn.to(target_device_str) # Ensure loss module is on device.
       
       # Tensors should also be moved to the device:
       # input_data = torch.randn(100, 1, 1).to(target_device_str)
       # output = model(input_data)
   ```

   This allows models and computations to run on the specified Intel GPU, potentially accelerating training and inference tasks. For NVIDIA GPUs, use `'cuda:0'`, and for CPU, use `'cpu'`.
```

## API Changes for Device Support

The following classes and functions were updated to include a `device` parameter. This parameter is a string that specifies the computation device (e.g., 'cpu', 'cuda:0', 'xpu:0') and typically defaults to 'cpu'.

**In `networks.py`:**
- `SimpleRNN(input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1, bias_fl=True, num_layers=1, device='cpu')`
- `GatedConvNet(channels=8, blocks=2, layers=9, dilation_growth=2, kernel_size=3, RNN_Input=True, device='cpu')`
- `RecNet(blocks=None, skip=0, device='cpu')`
- `BasicRNNBlock(params, device='cpu')`
- `load_model(model_data, device='cpu')`

**In `training.py`:**
- `ESRLoss(device='cpu')`
- `DCLoss(device='cpu')`
- `MultiSpecLoss(fft_sizes=(2048, 1024, 512, 256, 128), device='cpu')`
- `SpecLoss(fft_size=512, hop_size=128, device='cpu')`
- `PreEmph(filter_cfs, low_pass=0, device='cpu')`
- `LossWrapper(losses, pre_filt=None, device='cpu')`
