# Changelog

## [Unreleased] - YYYY-MM-DD

### Added
- **Intel GPU Support (XPU):** CoreAudioML now supports Intel GPUs via Intel® Extension for PyTorch (IPEX).
  - Relevant classes and functions in `networks.py` and `training.py` have been updated to accept a `device` string parameter (e.g., `'cpu'`, `'cuda:0'`, `'xpu:0'`) to control hardware placement for computations.
  - Documentation updated in `README.md` with instructions for setting up the environment for Intel GPUs and using the new `device` parameter.

### Changed
- **API Modifications for Device Support:**
  The following classes and functions were updated to include a `device` parameter. This parameter is a string that specifies the computation device and typically defaults to `'cpu'`.

  **In `networks.py`:**
  - `SimpleRNN(input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1, bias_fl=True, num_layers=1, device='cpu')`
  - `GatedConvNet(channels=8, blocks=2, layers=9, dilation_growth=2, kernel_size=3, RNN_Input=True, device='cpu')`
  - `ResConvBlock1DCausalGated(chan_input, chan_output, dilation_growth, kernel_size, layers, device='cpu')` (Note: This is an internal class but was updated to support device propagation from `GatedConvNet`)
  - `ResConvLayer1DCausalGated(chan_input, chan_output, dilation, kernel_size, device='cpu')` (Note: This is an internal class but was updated to support device propagation)
  - `RecNet(blocks=None, skip=0, device='cpu')`
  - `BasicRNNBlock(params, device='cpu')`
  - `load_model(model_data, device='cpu')`
  - `legacy_load(legacy_data, device='cpu')`

  **In `training.py`:**
  - `ESRLoss(device='cpu')`
  - `DCLoss(device='cpu')`
  - `MultiSpecLoss(fft_sizes=(2048, 1024, 512, 256, 128), device='cpu')`
  - `SpecLoss(fft_size=512, hop_size=128, device='cpu')`
  - `PreEmph(filter_cfs, low_pass=0, device='cpu')`
  - `LossWrapper(losses, pre_filt=None, device='cpu')`
