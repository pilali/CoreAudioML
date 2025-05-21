import torch
import torch.nn as nn
import torch.nn.functional as F

# ESR loss calculates the Error-to-signal between the output/target
class ESRLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(ESRLoss, self).__init__()
        self.device = device
        # Epsilon is a scalar, can be used directly. If it were a tensor, it would need .to(self.device)
        self.epsilon = 0.00001 

    def forward(self, output, target):
        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        # Ensure epsilon is treated correctly with device, PyTorch handles scalar addition to tensor on device
        energy = torch.mean(torch.pow(target, 2)) + torch.tensor(self.epsilon, device=target.device)
        loss = torch.div(loss, energy)
        return loss


class DCLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(DCLoss, self).__init__()
        self.device = device
        self.epsilon = 0.00001

    def forward(self, output, target):
        loss = torch.pow(torch.add(torch.mean(target, 0), -torch.mean(output, 0)), 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + torch.tensor(self.epsilon, device=target.device)
        loss = torch.div(loss, energy)
        return loss

# ESR loss calculates the Error-to-signal between the output/target
class MultiSpecLoss(nn.Module):
    def __init__(self, fft_sizes=(2048, 1024, 512, 256, 128), device='cpu'):
        super(MultiSpecLoss, self).__init__()
        self.device = device
        self.epsilon = 0.00001 # This epsilon seems unused here, but kept for compatibility if logic changes
        self.fft_sizes = fft_sizes
        self.spec_loss = []
        for size in self.fft_sizes:
            hop = size//4
            self.spec_loss.append(SpecLoss(size, hop, device=self.device))

    def forward(self, output, target):
        output = output.squeeze()
        target = target.squeeze()
        total_loss = 0
        for item in self.spec_loss: # item is already a SpecLoss instance on the correct device
            total_loss += item(output, target)
        return total_loss/len(self.fft_sizes)

class SpecLoss(nn.Module):
    def __init__(self, fft_size=512, hop_size=128, device='cpu'):
        super(SpecLoss, self).__init__()
        self.device = device
        self.epsilon = 0.00001 
        self.fft_size = fft_size
        self.hop_size = hop_size

    def forward(self, output, target):
        # STFT output device is determined by the input tensor's device
        magx = torch.abs(torch.stft(output, n_fft=self.fft_size, hop_length=self.hop_size, return_complex=True))
        magy = torch.abs(torch.stft(target, n_fft=self.fft_size, hop_length=self.hop_size, return_complex=True))

        # Ensure epsilon tensor is on the same device as magx/magy for torch.where
        epsilon_tensor = torch.tensor([self.epsilon], device=output.device)
        logx = torch.log(torch.where(magx <= self.epsilon, epsilon_tensor, magx))
        logy = torch.log(torch.where(magy <= self.epsilon, epsilon_tensor, magy))

        return F.l1_loss(magx, magy) + F.l1_loss(logx, logy)


# PreEmph is a class that applies an FIR pre-emphasis filter to the signal, the filter coefficients are in the
# filter_cfs argument, and lp is a flag that also applies a low pass filter
# Only supported for single-channel!
class PreEmph(nn.Module):
    def __init__(self, filter_cfs, low_pass=0, device='cpu'):
        super(PreEmph, self).__init__()
        self.device = device
        self.epsilon = 0.00001 # Seems unused in this class, but kept for compatibility
        self.zPad = len(filter_cfs) - 1

        self.conv_filter = nn.Conv1d(1, 1, 2, bias=False)
        # Move filter weights to the specified device
        self.conv_filter.weight.data = torch.tensor([[filter_cfs]], requires_grad=False, device=self.device)

        self.low_pass = low_pass
        if self.low_pass:
            self.lp_filter = nn.Conv1d(1, 1, 2, bias=False)
            # Move filter weights to the specified device
            self.lp_filter.weight.data = torch.tensor([[[0.85, 1]]], requires_grad=False, device=self.device)
        # Ensure the module itself is on the correct device
        self.to(device)


    def forward(self, output, target):
        # zero pad the input/target so the filtered signal is the same length
        # Ensure padding is on the same device as output/target
        output_padding = torch.zeros(self.zPad, output.shape[1], 1, device=output.device)
        target_padding = torch.zeros(self.zPad, target.shape[1], 1, device=target.device)
        output = torch.cat((output_padding, output))
        target = torch.cat((target_padding, target))
        
        # Apply pre-emph filter, permute because the dimension order is different for RNNs and Convs in pytorch...
        # .to(self.device) ensures filter and input are on the same device if not already
        output = self.conv_filter(output.permute(1, 2, 0))
        target = self.conv_filter(target.permute(1, 2, 0))

        if self.low_pass:
            output = self.lp_filter(output)
            target = self.lp_filter(target)

        return output.permute(2, 0, 1), target.permute(2, 0, 1)

class LossWrapper(nn.Module):
    def __init__(self, losses, pre_filt=None, device='cpu'):
        super(LossWrapper, self).__init__()
        self.device = device
        loss_dict = {'ESR': ESRLoss(device=self.device), 'DC': DCLoss(device=self.device)}
        
        # Store pre_filt module if it exists, to ensure it's part of this module's state (e.g. for .to(device) calls)
        self.pre_filt_module = None
        if pre_filt:
            self.pre_filt_module = PreEmph(pre_filt, device=self.device)
            # Original ESR (on self.device) will be used with pre-filtered output/target
            esr_loss_on_device = loss_dict['ESR'] 
            loss_dict['ESRPre'] = lambda output, target: esr_loss_on_device.forward(*self.pre_filt_module(output, target))
        
        loss_functions_list = []
        loss_factors_list = []

        for key, value in losses.items():
            loss_functions_list.append(loss_dict[key])
            loss_factors_list.append(value)

        self.loss_functions = tuple(loss_functions_list)
        
        if not loss_factors_list: # Handle empty losses dict
            self.loss_factors = torch.tensor([], device=self.device)
        elif all(isinstance(x, (int, float)) for x in loss_factors_list): # Ensure all are numbers
             self.loss_factors = torch.tensor(loss_factors_list, device=self.device)
        else: # Fallback or error if mixed types, for now, assume they are numbers or default to ones
            print("Warning: loss_factors contained non-numeric types or was empty. Defaulting to ones or empty tensor.")
            self.loss_factors = torch.ones(len(self.loss_functions), device=self.device) if self.loss_functions else torch.tensor([], device=self.device)


    def forward(self, output, target):
        loss = torch.tensor(0.0, device=output.device) # Initialize loss tensor on the output device
        for i, loss_fn in enumerate(self.loss_functions):
            # Ensure loss_factors[i] is a scalar or compatible tensor for multiplication
            factor = self.loss_factors[i].to(output.device) if self.loss_factors.numel() > 0 else torch.tensor(1.0, device=output.device)
            loss += torch.mul(loss_fn(output, target), factor)
        return loss


class TrainTrack(dict):
    def __init__(self):
        self.update({'current_epoch': 0, 'training_losses': [], 'validation_losses': [], 'train_av_time': 0.0,
                     'val_av_time': 0.0, 'total_time': 0.0, 'best_val_loss': 1e12, 'test_loss': 0})

    def restore_data(self, training_info):
        self.update(training_info)

    def train_epoch_update(self, loss, ep_st_time, ep_end_time, init_time, current_ep):
        if self['train_av_time']:
            self['train_av_time'] = (self['train_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['train_av_time'] = ep_end_time - ep_st_time
        self['training_losses'].append(loss)
        self['current_epoch'] = current_ep
        self['total_time'] += ((init_time + ep_end_time - ep_st_time)/3600)

    def val_epoch_update(self, loss, ep_st_time, ep_end_time):
        if self['val_av_time']:
            self['val_av_time'] = (self['val_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['val_av_time'] = ep_end_time - ep_st_time
        self['validation_losses'].append(loss)
        if loss < self['best_val_loss']:
            self['best_val_loss'] = loss
