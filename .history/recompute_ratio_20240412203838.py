import torch
from torch import nn
from functools import partial
import pdb

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w, scales


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w, scales


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.reshape(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t, scales


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.contiguous().view(-1, t_shape[-1])  ##contiguous added
    scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t, scales


class W8A8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', weight_quant='per_tensor', quantize_output=False): ## weight_quant added
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_quant_name='per_tensor'

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x, _ = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)    
        if self.output_quant_name== "None":
            q_y = self.output_quant(y)
        else:
            q_y, _ = self.output_quant(y)    
        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, weight_quant=weight_quant, quantize_output=quantize_output)
        if weight_quant == 'per_channel':
            new_module.weight, _ = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8)  # use 8-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight, _ = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias    
        return new_module

    def __repr__(self):
        return f'W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'



class NoisyW8A8Linear(W8A8Linear):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', quantize_output=False, err_prob=0.0, accumulation_bitw=32):
        super().__init__(in_features,out_features,bias,act_quant,quantize_output)
        assert isinstance(err_prob, list) or isinstance(err_prob, float)
        self.err_prob=err_prob
        self.accumulation_bitw = accumulation_bitw
        self.w_scales=None

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x
    
    @torch.no_grad()
    def inject_error(self,y, w_scales, a_scales, err_prob):
        y_not_quantized=y
        # y_div_a_scales=y/a_scales
        # result = y_div_a_scales/w_scales.view(1,1,-1)
        result=y.to(torch.float32)/(w_scales*a_scales)  ## integer of y
        result=result.round().to(torch.int32)
        result_injected=result.clone()

        flip_bit=torch.randint_like(result,0,31, device='cuda')
        err=torch.pow(2,flip_bit)
        prob_tensor=torch.full(result.shape, err_prob).to(result.device)
        mask=torch.bernoulli(prob_tensor).bool().to(result.device)

        pdb.set_trace()

        result_injected[mask]=torch.bitwise_xor(result[mask],err[mask])
        
        # result_injected=result_injected.to(torch.float32)*a_scales*w_scales.view(1,1,-1)
        result_injected=result_injected.to(torch.float32)*a_scales*w_scales
        result_injected=result_injected.to(y.dtype)
        y_not_quantized[mask]=result_injected[mask]

        # if not (y_not_quantized == y).all():
        #     import pdb; pdb.set_trace()
        return y_not_quantized

    @torch.no_grad()
    def forward(self, x):  
        q_x, x_scales = self.act_quant(x) ## q_x is multiplied by scale
        y = torch.functional.F.linear(q_x, self.weight, bias=None)
        y_for_quant= torch.functional.F.linear(q_x, self.weight, bias=self.bias)
        y_injected=self.inject_error(y, self.w_scales, x_scales, self.err_prob)
        if self.bias is not None:
            y_injected=y_injected + self.bias

        if self.output_quant_name== "None":
            q_y = self.output_quant(y_injected)  
            # clipping=y_for_quant.abs().max()

            # q_y=torch.where((q_y>clipping)|(q_y<-clipping), torch.zeros_like(q_y), q_y)
            # q_y = torch.clamp(q_y,-clipping,clipping) ## avoid overflowing of float16
        else:
            _, out_scale = self.output_quant(y_for_quant)  
            q_y=torch.clamp(torch.round(y_injected/out_scale),-127,127)*out_scale ## quant according to out_scale
        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False, err_prob=0.0,accumulation_width=32):
        assert isinstance(module, torch.nn.Linear)
        new_module = NoisyW8A8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, quantize_output=quantize_output,err_prob=err_prob,accumulation_bitw=accumulation_width)
        if weight_quant == 'per_channel':
            new_module.weight, new_module.w_scales = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8)  # use 8-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight, new_module.w_scales = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f'NoisyW8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, err_prob={self.err_prob})'


def severe


# hidden_states=torch.load('hidden_states.pt')
# ln_test=layer_norm_without_outlier(1)
# test=ln_test(hidden_states)
# ln=nn.LayerNorm(2048).to('cuda')
# gold=ln(hidden_states)

# pdb.set_trace()
tt=torch.nn.Linear(128,128,bias=True,device='cuda')
given_weight=torch.randn(2,128,128,device='cuda')
given_activation=torch.randn(2,128,128,device='cuda')
# given_bias=torch.randn(108)
# tt.weight.data=given_weight
# tt.bias.data=given_bias
data_in=torch.randn(128,128,device='cuda')
err_prob=1e-2
normal=W8A8Linear.from_float(tt,weight_quant='per_tensor',act_quant='per_tensor',quantize_output=True)
noisy=NoisyW8A8Linear.from_float(tt,weight_quant='per_tensor',act_quant='per_tensor',quantize_output=True, err_prob=err_prob)

out_normal=normal(data_in)
out_noisy=noisy(data_in)

# bmm_normal=W8A8BMM(act_quant='per_tensor',quantize_output=True)
# bmm_noisy=NoisyW8A8BMM(act_quant='per_tensor',quantize_output=True,err_prob=err_prob)

# out_bmm_normal=bmm_normal(given_weight,given_activation)
# out_bmm_noisy=bmm_noisy(given_weight,given_activation)
    
# data_in=torch.load('../input.pt')
# weight=torch.load('../weight.pt')
# tt=torch.nn.Linear(5120,5120,bias=False)
# tt.weight.data=weight
# err_prob=0.0
# normal=W8A8Linear.from_float(tt,weight_quant='per_tensor',act_quant='per_tensor',quantize_output=True)
# noisy=NoisyW8A8Linear.from_float(tt,weight_quant='per_tensor',act_quant='per_tensor',quantize_output=True, err_prob=err_prob)    
# out_normal=normal(data_in)
# out_noisy=noisy(data_in)



pdb.set_trace()
