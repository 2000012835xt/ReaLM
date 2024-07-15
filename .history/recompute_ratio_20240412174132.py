import torch
from torch import nn
from functools import partial
import pdb


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
        y_div_a_scales=y/a_scales
        result = y_div_a_scales/w_scales.view(1,1,-1)
        # result=y.to(torch.float32)/(w_scales*a_scales)  ## integer of y
        result=result.round().to(torch.int32)
        result_injected=result

        flip_bit=30
        err=torch.tensor([2**flip_bit],dtype=torch.int32).to(result.device)
        prob_tensor=torch.full(result.shape, err_prob).to(result.device)
        mask=torch.bernoulli(prob_tensor).bool().to(result.device)

        result_injected[mask]=torch.bitwise_xor(result[mask],err)
        
        result_injected=result_injected.to(torch.float32)*a_scales*w_scales.view(1,1,-1)
        # result_injected=result_injected.to(torch.float32)*a_scales*w_scales
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
