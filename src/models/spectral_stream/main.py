#!/usr/bin/env python3
"""
GPU-optimized version of SpectralStream pipeline. 
Key improvements:
 - vectorized per-pixel windows using tensor.unfold (GPU-friendly)
 - learned linear projection instead of allocating large zero tensors
 - tile batching and optional mixed precision
 - fewer CPU-GPU transfers
"""
import argparse
import os
import math
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# ---------------------------
# Vectorized sliding windows 
# Input: spectral tensor (B, H, W, Bands)
# Output: windows tensor (B, H, W, n_windows, window_size)
# ---------------------------

def sliding_windows_per_pixel_gpu(spectral: torch.Tensor, window_size: int, stride: int):
    # spectral: (B,H,W,Bands)
    B, H, W, Bands = spectral.shape
    if Bands < window_size:
        pad = window_size - Bands
        spectral = F.pad(spectral, (0, pad), "constant", 0.0)
        Bands = window_size

    # reshape to (B*H*W, Bands)
    flat = spectral.permute(0,1,2,3).contiguous().view(-1, Bands)  # (B*H*W, Bands)
    # use unfold on last dim -> (BHW, n_windows, window_size)
    n_windows = 1 + max(0, (Bands - window_size) // stride)
    if n_windows <= 0:
        # fallback: one window (0:window_size)
        w = flat[:, :window_size].unsqueeze(1)  # (BHW,1,ws)
    else:
        w = flat.unfold(dimension=1, size=window_size, step=stride)  # (BHW, n_windows, ws)

    # reshape to (B, H, W, n_windows, window_size)
    BHW = B*H*W
    nw = w.shape[1]
    ws = w.shape[2]
    w = w.view(B, H, W, nw, ws)
    return w  # (B,H,W,nw,ws)


# ---------------------------
# Mamba wrapper: use learned linear projection
# ---------------------------
class MambaWrapper(nn.Module):
    def __init__(self, window_size: int, model_dim: int):
        super().__init__()
        self.mamba = Mamba(
            d_model=model_dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.window_size = window_size
        self.model_dim = model_dim
        # replace zero-alloc trick with a small linear projection
        self.input_proj = nn.Linear(1, model_dim, bias=False)

    def forward(self, windows: torch.Tensor):
        """
        windows: (B,H,W,nw,ws)
        returns: (B,H,W,nw,model_dim)
        """
        B,H,W,nw,ws = windows.shape
        assert ws == self.window_size, f"Expected {self.window_size}, got {ws}"
        # merge dims -> (B*H*W*nw, ws, 1)
        x = windows.contiguous().view(-1, ws).unsqueeze(-1)  # (batch', ws, 1)
        # project along last dim using a linear layer applied per time-step
        # linear expects (..., in_features); we can flatten time dim and apply then reshape
        batch = x.shape[0]
        x_flat = x.view(-1, 1)  # (batch'*ws, 1)
        x_proj_flat = self.input_proj(x_flat)  # (batch'*ws, model_dim)
        x_proj = x_proj_flat.view(batch, ws, self.model_dim)  # (batch', ws, model_dim)
        # pass through mamba -> (batch', ws, model_dim)
        y = self.mamba(x_proj)
        # mean pool across sequence length
        y_pooled = y.mean(dim=1)  # (batch', model_dim)
        return y_pooled.view(B, H, W, nw, self.model_dim)


# ---------------------------
# Attention pooling 
# ---------------------------
class WindowAttentionPool(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, X: torch.Tensor):
        B,H,W,nw,D = X.shape
        scores = self.att_mlp(X.view(-1,D)).view(B,H,W,nw)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (X * weights).sum(dim=-2)


# ---------------------------
# Positional encodings
# ---------------------------
class Spectral1DPosEncoding(nn.Module):
    def __init__(self, D: int): super().__init__(); self.D=D
    def forward(self, lam: torch.Tensor):
        if lam.ndim!=4 or lam.shape[-1]!=1: lam=lam.reshape(*lam.shape[:3],1)
        B,H,W,_=lam.shape; lam=lam.reshape(-1,1)
        dims=torch.arange(self.D//2,dtype=lam.dtype,device=lam.device)
        div=10000**(2*dims/self.D); ang=lam/div
        pe=torch.cat([torch.sin(ang),torch.cos(ang)],dim=1)[:,:self.D]
        return pe.reshape(B,H,W,self.D)

class Spatial2DPosEncoding(nn.Module):
    def __init__(self,D:int): super().__init__(); self.D=D
    def forward(self,i:torch.Tensor,j:torch.Tensor):
        i_flat=i.reshape(-1).float().unsqueeze(1); j_flat=j.reshape(-1).float().unsqueeze(1)
        d_half=max(1,self.D//2); dims=torch.arange(d_half,dtype=i_flat.dtype,device=i_flat.device)
        div=10000**(2*dims/(2*d_half)); ai=i_flat/div; aj=j_flat/div
        pe=torch.cat([torch.sin(ai),torch.cos(ai),torch.sin(aj),torch.cos(aj)],dim=1)[:,:self.D]
        return pe.reshape(*i.shape,self.D)


# ---------------------------
# SpectralStream with Mamba wrappers
# ---------------------------
class SpectralStreamMamba(nn.Module):
    def __init__(self, band_count, window_sizes=[8,16,32],
                 model_dim=64, token_dim=128, out_dim=128, device='cuda'):
        super().__init__()
        self.out_dim=out_dim; self.token_dim=token_dim
        self.blocks=nn.ModuleDict(); self.poolers=nn.ModuleDict()
        for ws in window_sizes:
            self.blocks[f"w{ws}"]=MambaWrapper(ws, model_dim)
            self.poolers[f"w{ws}"]=WindowAttentionPool(model_dim)
        self.proj=nn.Linear(len(window_sizes)*model_dim,out_dim)
        self.token_proj=nn.Linear(out_dim,token_dim)
        self.spec_pe=Spectral1DPosEncoding(token_dim//2)
        self.spat_pe=Spatial2DPosEncoding(token_dim//2)
        self.head=nn.Sequential(nn.Linear(token_dim,token_dim),nn.ReLU(),nn.Linear(token_dim,out_dim))

    def forward(self,x):
        # x: (B,H,W,Bands)
        B,H,W,Bands=x.shape; per_scale=[]
        for ws_key in self.blocks:
            ws = int(ws_key[1:])
            stride=max(1,ws//2)
            windows = sliding_windows_per_pixel_gpu(x, ws, stride)  # vectorized
            feats = self.blocks[ws_key](windows)  # (B,H,W,nw,model_dim)
            pooled = self.poolers[ws_key](feats)  # (B,H,W,model_dim)
            per_scale.append(pooled)
        concat = torch.cat(per_scale, dim=-1)  # (B,H,W, len(ws)*model_dim)
        F_spectral = self.proj(concat)
        tokens = self.token_proj(F_spectral)
        device=x.device
        spec_pe = self.spec_pe(torch.full((B,H,W,1), float(Bands)/2.0, device=device))
        ii=torch.arange(H,device=device).view(1,H,1).expand(B,H,W)
        jj=torch.arange(W,device=device).view(1,1,W).expand(B,H,W)
        spat_pe = self.spat_pe(ii,jj)
        T_tilde = tokens + torch.cat([spec_pe, spat_pe], dim=-1)
        out = self.head(T_tilde)
        return F_spectral, T_tilde, out


# ---------------------------
# Data helpers 
# ---------------------------
def read_cube(path:str)->np.ndarray:
    arr=tiff.imread(path); arr=np.asarray(arr)
    if arr.ndim==3:
        b0,b1,b2=arr.shape
        if b0<=512 and b0<b1 and b0<b2: cube=np.moveaxis(arr,0,-1)
        elif b2<=512 and b2<b0 and b2<b1: cube=arr
        else: cube=arr if arr.shape[2]<=300 else np.moveaxis(arr,0,-1)
    else: raise RuntimeError(f"Bad tif shape {arr.shape}")
    return cube.astype(np.float32)

def normalize_cube(cube:np.ndarray)->np.ndarray:
    cube=cube.astype(np.float32); mx=cube.max()
    return cube/mx if mx>1.1 else cube

def tiles_for_shape(H,W,tile,overlap):
    step=tile-overlap; ys=list(range(0,H,step)); xs=list(range(0,W,step))
    if ys[-1]+tile<H: ys[-1]=H-tile
    if xs[-1]+tile<W: xs[-1]=W-tile
    return [(y,x) for y in ys for x in xs]


# ---------------------------
# Process tiles with batching and optional AMP
# ---------------------------
def process_tiles_batched(cube, model, device, tile=256, overlap=32, batch_tiles=4, use_amp=False):
    H,W,B=cube.shape; out_dim=model.out_dim; token_dim=model.token_dim
    sum_feats=np.zeros((H,W,out_dim),dtype=np.float32)
    sum_tokens=np.zeros((H,W,token_dim),dtype=np.float32)
    counts=np.zeros((H,W),dtype=np.float32)
    coords=tiles_for_shape(H,W,tile,overlap)
    model.eval()
    # process coords in batches
    with torch.no_grad():
        for i in range(0, len(coords), batch_tiles):
            batch = coords[i:i+batch_tiles]
            patches = []
            coords_actual = []
            for (y,x) in batch:
                y2,x2=y+tile,x+tile; patch=cube[y:y2,x:x2,:]
                ph,pw=patch.shape[:2]
                if ph!=tile or pw!=tile:
                    pad=((0,tile-ph),(0,tile-pw),(0,0)); patch=np.pad(patch,pad)
                patches.append(patch)
                coords_actual.append((y,x,ph,pw))
            tpatch = np.stack(patches, axis=0)  # (BT, tile, tile, B)
            tpatch = torch.from_numpy(tpatch).to(device)
            # optionally use AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    Fs_batch, Tt_batch, _ = model(tpatch)
            else:
                Fs_batch, Tt_batch, _ = model(tpatch)
            Fs_batch = Fs_batch.cpu().numpy()
            Tt_batch = Tt_batch.cpu().numpy()
            # scatter results back
            for idx, (y,x,ph,pw) in enumerate(coords_actual):
                Fs = Fs_batch[idx][:ph,:pw,:]
                Tt = Tt_batch[idx][:ph,:pw,:]
                sum_feats[y:y+ph, x:x+pw, :] += Fs
                sum_tokens[y:y+ph, x:x+pw, :] += Tt
                counts[y:y+ph, x:x+pw] += 1.0
    counts[counts==0]=1.0
    return sum_feats/counts[...,None], sum_tokens/counts[...,None]


# ---------------------------
# Save
# ---------------------------
def save_features(out_path,F_spectral,T_tokens,meta=None):
    base,_=os.path.splitext(out_path)
    np.savez_compressed(base+".npz",F_spectral=F_spectral,T_tokens=T_tokens,meta=meta or {})
    print("Saved:",base+".npz")


# ---------------------------
# Main CLI
# ---------------------------
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cube",type=str,required=True)
    p.add_argument("--out",type=str,required=True)
    p.add_argument("--tile",type=int,default=256)
    p.add_argument("--overlap",type=int,default=32)
    p.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model_dim",type=int,default=64)
    p.add_argument("--token_dim",type=int,default=128)
    p.add_argument("--out_dim",type=int,default=128)
    p.add_argument("--batch_tiles",type=int,default=2, help="Number of tiles processed at once (increase if you have memory)")
    p.add_argument("--amp", action='store_true', help="Use torch.cuda.amp for mixed precision")
    args=p.parse_args()

    cube = normalize_cube(read_cube(args.cube))
    H,W,B = cube.shape
    print("Cube", cube.shape)
    device = torch.device(args.device)
    ws = [8,16,32] if B>=32 else [max(4,B//4)]
    model = SpectralStreamMamba(B, window_sizes=ws,
                                model_dim=args.model_dim,
                                token_dim=args.token_dim,
                                out_dim=args.out_dim).to(device)

    Fs, Tt = process_tiles_batched(cube, model, device, tile=args.tile, overlap=args.overlap, batch_tiles=args.batch_tiles, use_amp=args.amp)
    print("Output shapes:", Fs.shape, Tt.shape)
    save_features(args.out, Fs, Tt, meta={"source": args.cube, "bands": B})

if __name__=="__main__":
    main()
