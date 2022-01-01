import sys
import torch
import torch.nn.functional as F
from packaging import version
import numpy as np

def cosineD(p, z, softmax_val=None):
    return (1-(p*z).sum(dim=1)).mean()
    
def mseLoss(x, y):
    return (torch.sqrt( torch.sum( (x-y)**2 ) ) ).mean()

def normalize(x, dim):
    x_norm = torch.norm(x, p=2, dim=dim, keepdim=True) + sys.float_info.epsilon #p=2:Frobenius norm
    x = torch.div(x,x_norm)
    return x
    
def centerL2Norm(x,dim):
    x = x- x.mean(dim=dim, keepdim=True)
    x_norm = torch.norm(x, p=2, dim=dim, keepdim=True) + sys.float_info.epsilon #p=2:Frobenius norm
    x = torch.div(x,x_norm)
    return x

def get_zero_tensor(input, device="cuda"):
    if "cuda" in device:
        zero_tensor = torch.cuda.FloatTensor(1).fill_(0)
    else:
        zero_tensor = torch.FloatTensor(1).fill_(0)

    zero_tensor.requires_grad_(False)
    return zero_tensor.expand_as(input)

def WTAStyleLoss(args, style, output, useDetach=False, iscenterL2Norm=True, kernel_size=3, stylePatchSample=False):
    """  must modify temperature """
    temp = args.nns_temp
    mask_type = torch.bool
    if not stylePatchSample:
        b,d,h,w = style.shape # b=1

        if useDetach:
            style = style.detach()
        style = F.unfold(style, kernel_size=kernel_size, padding=int(kernel_size//2))
        output = F.unfold(output, kernel_size=kernel_size, padding=int(kernel_size//2))
        
        style = style.view(b,-1,h*w) # (b, d, hw)
        output = output.view(b,-1,h*w) # (b, d, hw)
        
        style = style.view(-1,h*w).permute(1,0).contiguous()   # (hw, d)
        output = output.view(-1,h*w).permute(1,0).contiguous() # (hw, d)

        style_clone = style.clone()
        output_clone = output.clone()

        if iscenterL2Norm:
            style = centerL2Norm(style, dim=1) # l2-normalize
            output = centerL2Norm(output, dim=1) # l2-normalize
        else:
            style  = normalize(style, dim=1) # l2-normalize
            output = normalize(output, dim=1) # l2-normalize

        corr = torch.mm(output, style.permute(1,0).contiguous() ) + sys.float_info.epsilon
        softmax_val, argmax = F.softmax(corr/temp, dim=-1).max(dim=-1) # (num_patch, 1)
        #l_pos, argmax = corr.max(dim=-1) # [0], [1]


        """ here """
        batch_range = torch.arange(1).view(1,-1)
        x, y = output, style[argmax, :] # num_patch , d

        loss = 0.
        loss += cosineD(x, y)

    else:
        num_patches, d = style.shape
        
        style_clone = style.clone()
        output_clone = output.clone()

        if useDetach:
            style = style.detach()
        if iscenterL2Norm:
            style = centerL2Norm(style, dim=1) # l2-normalize
            output = centerL2Norm(output, dim=1) # l2-normalize
        else:
            style  = normalize(style, dim=1) # l2-normalize
            output = normalize(output, dim=1) # l2-normalize

        corr = torch.mm(output, style.permute(1,0).contiguous() ) + sys.float_info.epsilon

        softmax_val,argmax = F.softmax(corr/temp, dim=-1).max(dim=-1) # (num_patch, 1)
        #l_pos, argmax = corr.max(dim=-1) # [0], [1]

        """ here """
        batch_range = torch.arange(1).view(1,-1)
        x, y = output, style[argmax, :] # num_patch , d

        """ option 2 """
        loss = 0.
        loss += cosineD(x, y)

        return loss


def patchLocalizer(img_shape, num_patches=64*32):
    """
    content.shape: (b,d,h,w)
    output.shape: (b,d,h,w)
    """
    num_patches = num_patches
    
    b,d,h,w = img_shape
    
    patch_range = h*w
    patch_id = torch.randperm(patch_range)
    patch_id = patch_id[ :int(min(num_patches, patch_id.shape[0])) ] # (#patches)

    return patch_id

def patchSampler(args, content, output, patch_id, is_unfold=False, kernel_size=3, corr_feat=None):
    """
    content.shape: (b,d,h,w)
    output.shape: (b,d,h,w)
    """
    
    b,d,h,w = content.shape

    if not is_unfold:
        content_view = content.view(b,d,h*w).permute(0,2,1).contiguous().flatten(0,1) # (b*h*w, d)
        output_view = output.view(b,d,h*w).permute(0,2,1).contiguous().flatten(0,1) # (b*h*w, d)
    else:
        content = F.unfold(content, kernel_size=kernel_size, padding=int(kernel_size//2))
        output = F.unfold(output, kernel_size=kernel_size, padding=int(kernel_size//2))

        content_view = content.view(b,-1,h*w).permute(0,2,1).flatten(0,1) # style.view(b,-1,h*w).shape=(b, d, hw) // (b*h*w, d)
        output_view = output.view(b,-1,h*w).permute(0,2,1).flatten(0,1) # (b*h*w, d)

    content_sample = content_view[patch_id, :]
    output_sample = output_view[patch_id, :]
    
    return [content_sample, output_sample] #(#patches, d)

def NCELoss(args, content, output, useDetach=False, iscenterL2Norm=True):
    mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
    temp = args.nce_temp

    if useDetach:
        content = content.detach()
    num_patch, d = content.shape

    if iscenterL2Norm:
        content = centerL2Norm(content, dim=-1) # l2-normalize
        output  = centerL2Norm(output, dim=-1) # l2-normalize
    else:
        content = normalize(content, dim=-1) # l2-normalize
        output  = normalize(output, dim=-1) # l2-normalize
    

    """
        corr = torch.mm(output, style.permute(1,0).contiguous() ) + sys.float_info.epsilon
        l_pos, argmax = corr.max(dim=-1) # [0], [1]

        batch_range = torch.arange(1).view(1,-1)
        x, y = output, style[argmax, :] # num_patch , d
    """

    l_pos = (content * output).sum(dim=-1)[:, None] #(#patches,1)
    l_neg = torch.mm(output, content.permute(1,0).contiguous() ) # (#patches, #patches)
    
    identity_matrix = torch.eye(num_patch, dtype=mask_dtype).to(args.device) # (num_patch, num_patch)

    l_neg_mask = l_neg.masked_fill(identity_matrix, -10.0)
    logits = torch.cat( (l_pos, l_neg_mask), dim=1 )/temp # (num_patch, num_patch+1)

    predictions = logits#.flatten(0,1) # (num_patch,num_patch+1)
    targets = torch.zeros(num_patch, dtype=torch.long).to(args.device) #(num_patch)

    loss = F.cross_entropy(predictions, targets)

    return loss
