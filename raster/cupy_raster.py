from __future__ import annotations

import math
import numpy as np
import torch

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

Tensor = torch.Tensor

CUDA_SRC = r'''
extern "C" __global__ void ewa_depth(
    const float* p, const float* tau, const float* grad, const float* invcov, const int* radius,
    const unsigned char* radmask, const int* surface_ids,
    const int n, const int H, const int W, const float chi2,
    float* depth, unsigned long long* first_key,
    unsigned long long* touches, unsigned long long* depth_only_touches) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float px = p[2*i+0], py = p[2*i+1];
    float t0 = tau[i];
    float gx = grad[2*i+0], gy = grad[2*i+1];
    float a = invcov[4*i+0], b = invcov[4*i+1], c = invcov[4*i+2], d = invcov[4*i+3];
    int r = radius[i];
    int cx = __float2int_rn(px), cy = __float2int_rn(py);
    for (int oy=-r; oy<=r; ++oy) for (int ox=-r; ox<=r; ++ox) {
        int x = cx + ox, y = cy + oy;
        if (x < 0 || x >= W || y < 0 || y >= H) continue;
        float dx = ((float)x) - px, dy = ((float)y) - py;
        float qx = a*dx + b*dy, qy = c*dx + d*dy;
        float q = dx*qx + dy*qy;
        if (q > chi2) continue;
        float z = t0 + gx*dx + gy*dy;
        if (!(z > 0.0f) || !isfinite(z)) continue;
        atomicMin((unsigned int*)&depth[y*W+x], __float_as_uint(z));
        unsigned long long key=((unsigned long long)__float_as_uint(z)<<32) | (unsigned int)surface_ids[i];
        atomicMin(&first_key[y*W+x], key);
        atomicAdd(touches, 1ULL);
        if (radmask[i] == 0) atomicAdd(depth_only_touches, 1ULL);
    }
}

extern "C" __global__ void ewa_color(
    const float* p, const float* tau, const float* grad, const float* invcov, const int* radius,
    const float* color, const unsigned char* radmask, const int* surface_ids,
    const int n, const int H, const int W, const float chi2, const float depth_eps,
    const float* depth, const unsigned long long* first_key, float* wsum, float* rgb) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n || radmask[i] == 0) return;
    float px = p[2*i+0], py = p[2*i+1];
    float t0 = tau[i];
    float gx = grad[2*i+0], gy = grad[2*i+1];
    float a = invcov[4*i+0], b = invcov[4*i+1], c = invcov[4*i+2], d = invcov[4*i+3];
    int r = radius[i];
    int cx = __float2int_rn(px), cy = __float2int_rn(py);
    for (int oy=-r; oy<=r; ++oy) for (int ox=-r; ox<=r; ++ox) {
        int x = cx + ox, y = cy + oy;
        if (x < 0 || x >= W || y < 0 || y >= H) continue;
        float dx = ((float)x) - px, dy = ((float)y) - py;
        float qx = a*dx + b*dy, qy = c*dx + d*dy;
        float q = dx*qx + dy*qy;
        if (q > chi2) continue;
        float z = t0 + gx*dx + gy*dy;
        int id = y*W+x;
        if (!(z > 0.0f) || !isfinite(z) || z > depth[id] + depth_eps) continue;
        if ((unsigned int)surface_ids[i] != (unsigned int)(first_key[id] & 0xffffffffULL)) continue;
        float w = expf(-0.5f*q);
        atomicAdd(&wsum[id], w);
        atomicAdd(&rgb[3*id+0], w*color[3*i+0]);
        atomicAdd(&rgb[3*id+1], w*color[3*i+1]);
        atomicAdd(&rgb[3*id+2], w*color[3*i+2]);
    }
}

__device__ inline bool bary(
    float px, float py,
    float ax, float ay, float bx, float by, float cx, float cy,
    float* w0, float* w1, float* w2) {
    float v0x=bx-ax, v0y=by-ay, v1x=cx-ax, v1y=cy-ay, v2x=px-ax, v2y=py-ay;
    float den = v0x*v1y - v1x*v0y;
    if (fabsf(den) < 1e-12f) return false;
    float u = (v2x*v1y - v1x*v2y) / den;
    float v = (v0x*v2y - v2x*v0y) / den;
    float w = 1.0f - u - v;
    *w0=w; *w1=u; *w2=v;
    return (u >= -1e-6f && v >= -1e-6f && w >= -1e-6f);
}

extern "C" __global__ void tri_depth(
    const float* p, const float* tau, const unsigned char* radmask, const int* surface_ids,
    const int ntri, const int H, const int W, float* depth,
    unsigned long long* first_key, unsigned long long* touches,
    unsigned long long* depth_only_touches) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ntri) return;
    float ax=p[6*i+0], ay=p[6*i+1], bx=p[6*i+2], by=p[6*i+3], cx=p[6*i+4], cy=p[6*i+5];
    int xmin=max(0, (int)floorf(fminf(ax,fminf(bx,cx))));
    int xmax=min(W-1, (int)ceilf(fmaxf(ax,fmaxf(bx,cx))));
    int ymin=max(0, (int)floorf(fminf(ay,fminf(by,cy))));
    int ymax=min(H-1, (int)ceilf(fmaxf(ay,fmaxf(by,cy))));
    for (int y=ymin; y<=ymax; ++y) for (int x=xmin; x<=xmax; ++x) {
        float w0,w1,w2;
        if (!bary((float)x,(float)y,ax,ay,bx,by,cx,cy,&w0,&w1,&w2)) continue;
        float z=w0*tau[3*i+0]+w1*tau[3*i+1]+w2*tau[3*i+2];
        if (!(z > 0.0f) || !isfinite(z)) continue;
        atomicMin((unsigned int*)&depth[y*W+x], __float_as_uint(z));
        unsigned long long key=((unsigned long long)__float_as_uint(z)<<32) | (unsigned int)surface_ids[i];
        atomicMin(&first_key[y*W+x], key);
        atomicAdd(touches, 1ULL);
        if (radmask[i] == 0) atomicAdd(depth_only_touches, 1ULL);
    }
}

extern "C" __global__ void tri_color(
    const float* p, const float* tau, const float* color, const unsigned char* radmask,
    const int* surface_ids,
    const int ntri, const int H, const int W, const float depth_eps, const float* depth,
    const unsigned long long* first_key, float* wsum, float* rgb) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ntri || radmask[i] == 0) return;
    float ax=p[6*i+0], ay=p[6*i+1], bx=p[6*i+2], by=p[6*i+3], cx=p[6*i+4], cy=p[6*i+5];
    int xmin=max(0, (int)floorf(fminf(ax,fminf(bx,cx))));
    int xmax=min(W-1, (int)ceilf(fmaxf(ax,fmaxf(bx,cx))));
    int ymin=max(0, (int)floorf(fminf(ay,fminf(by,cy))));
    int ymax=min(H-1, (int)ceilf(fmaxf(ay,fmaxf(by,cy))));
    for (int y=ymin; y<=ymax; ++y) for (int x=xmin; x<=xmax; ++x) {
        float w0,w1,w2;
        if (!bary((float)x,(float)y,ax,ay,bx,by,cx,cy,&w0,&w1,&w2)) continue;
        float z=w0*tau[3*i+0]+w1*tau[3*i+1]+w2*tau[3*i+2];
        int id=y*W+x;
        if (!(z > 0.0f) || !isfinite(z) || z > depth[id]+depth_eps) continue;
        if ((unsigned int)surface_ids[i] != (unsigned int)(first_key[id] & 0xffffffffULL)) continue;
        float rr=w0*color[9*i+0]+w1*color[9*i+3]+w2*color[9*i+6];
        float gg=w0*color[9*i+1]+w1*color[9*i+4]+w2*color[9*i+7];
        float bb=w0*color[9*i+2]+w1*color[9*i+5]+w2*color[9*i+8];
        atomicAdd(&wsum[id], 1.0f);
        atomicAdd(&rgb[3*id+0], rr);
        atomicAdd(&rgb[3*id+1], gg);
        atomicAdd(&rgb[3*id+2], bb);
    }
}

extern "C" __global__ void extract_id(const unsigned long long* first_key, int n, int* first_id) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long key=first_key[i];
    first_id[i] = key == 0xffffffffffffffffULL ? -1 : (int)(key & 0xffffffffULL);
}
'''

_MODULE = None


def available() -> bool:
    return cp is not None and torch.cuda.is_available()


def _module():
    global _MODULE
    if _MODULE is None:
        if cp is None:
            raise RuntimeError("CuPy is not installed")
        _MODULE = cp.RawModule(code=CUDA_SRC, options=("--std=c++11",))
    return _MODULE


def _cp(t: Tensor):
    return cp.from_dlpack(t.contiguous())


def _launch(kernel, blocks, threads, args, device):
    # DLPack shares Torch storage. Keep the launch ordered with both the Torch
    # producers and the Torch consumers of those buffers.
    with cp.cuda.Device(device.index or torch.cuda.current_device()):
        with cp.cuda.Stream.from_external(torch.cuda.current_stream(device)):
            kernel((blocks,), (threads,), args)


def rasterize_ewa_cupy(
    p: Tensor, tau: Tensor, color: Tensor, cov: Tensor, depth_grad: Tensor, radiance_mask: Tensor,
    height: int, width: int, mirror_mask: Tensor | None = None,
    environment_rgb=(0.03,0.04,0.06), max_radius: int = 8, depth_eps: float = 1e-3, chi2: float = 9.0,
    surface_ids: Tensor | None = None,
):
    if not available() or p.device.type != "cuda":
        raise RuntimeError("CuPy CUDA backend unavailable")
    dtype = p.dtype
    if dtype != torch.float32:
        raise ValueError("CuPy rasterizer currently requires float32")
    eye = torch.eye(2, device=p.device, dtype=dtype)[None]
    invcov = torch.linalg.inv(cov + 1e-6 * eye)
    evals = torch.linalg.eigvalsh(cov).clamp_min(1e-8)
    radius = torch.ceil(3.0 * torch.sqrt(evals[..., -1])).to(torch.int32).clamp(1, max_radius)
    depth = torch.full((height*width,), float("inf"), device=p.device, dtype=torch.float32)
    wsum = torch.zeros((height*width,), device=p.device, dtype=torch.float32)
    rgb = torch.zeros((height*width,3), device=p.device, dtype=torch.float32)
    touches = torch.zeros((1,), device=p.device, dtype=torch.int64)
    depth_only_touches = torch.zeros((1,), device=p.device, dtype=torch.int64)
    first_key = torch.full((height*width,), -1, device=p.device, dtype=torch.int64)
    first_id = torch.full((height*width,), -1, device=p.device, dtype=torch.int32)
    if surface_ids is None:
        surface_ids = torch.zeros((p.shape[0],), device=p.device, dtype=torch.int32)
    else:
        surface_ids = surface_ids.to(device=p.device, dtype=torch.int32)
    rad = radiance_mask.to(torch.uint8)
    mod = _module()
    threads=128; blocks=(p.shape[0]+threads-1)//threads
    if blocks:
        shape_args = (np.int32(p.shape[0]), np.int32(height), np.int32(width))
        _launch(mod.get_function("ewa_depth"),blocks,threads,(_cp(p),_cp(tau),_cp(depth_grad),_cp(invcov),_cp(radius),_cp(rad),_cp(surface_ids),*shape_args,np.float32(chi2),_cp(depth),_cp(first_key),_cp(touches),_cp(depth_only_touches)),p.device)
        _launch(mod.get_function("ewa_color"),blocks,threads,(_cp(p),_cp(tau),_cp(depth_grad),_cp(invcov),_cp(radius),_cp(color),_cp(rad),_cp(surface_ids),*shape_args,np.float32(chi2),np.float32(depth_eps),_cp(depth),_cp(first_key),_cp(wsum),_cp(rgb)),p.device)
        n_pixels=height*width
        _launch(mod.get_function("extract_id"),(n_pixels+threads-1)//threads,threads,(_cp(first_key),np.int32(n_pixels),_cp(first_id)),p.device)
    env=torch.tensor(environment_rgb,device=p.device,dtype=torch.float32)
    img=env.expand(height*width,3).clone()
    finite=torch.isfinite(depth); has=wsum>1e-8
    img[finite]=0.0
    img[has]=rgb[has]/wsum[has,None]
    img=img.reshape(height,width,3); depth=depth.reshape(height,width)
    first_id=first_id.reshape(height,width)
    if mirror_mask is not None:
        img=torch.where(mirror_mask[...,None],img,torch.zeros_like(img))
        depth=torch.where(mirror_mask,depth,torch.full_like(depth,float("inf")))
        first_id=torch.where(mirror_mask,first_id,-1)
    return img, depth, first_id, {"pixel_touches": int(touches.item()), "depth_only_pixel_touches": int(depth_only_touches.item())}


def rasterize_triangles_cupy(
    tri_p: Tensor, tri_tau: Tensor, tri_color: Tensor, tri_radiance_mask: Tensor,
    height: int, width: int, mirror_mask: Tensor | None = None,
    environment_rgb=(0.03,0.04,0.06), depth_eps: float = 1e-3,
    surface_ids: Tensor | None = None,
):
    if not available() or tri_p.device.type != "cuda":
        raise RuntimeError("CuPy CUDA backend unavailable")
    if tri_p.dtype != torch.float32:
        raise ValueError("CuPy rasterizer currently requires float32")
    depth=torch.full((height*width,),float("inf"),device=tri_p.device,dtype=torch.float32)
    wsum=torch.zeros((height*width,),device=tri_p.device,dtype=torch.float32)
    rgb=torch.zeros((height*width,3),device=tri_p.device,dtype=torch.float32)
    touches=torch.zeros((1,),device=tri_p.device,dtype=torch.int64)
    depth_only_touches=torch.zeros((1,),device=tri_p.device,dtype=torch.int64)
    first_key=torch.full((height*width,),-1,device=tri_p.device,dtype=torch.int64)
    first_id=torch.full((height*width,),-1,device=tri_p.device,dtype=torch.int32)
    if surface_ids is None:
        surface_ids=torch.zeros((tri_p.shape[0],),device=tri_p.device,dtype=torch.int32)
    else:
        surface_ids=surface_ids.to(device=tri_p.device,dtype=torch.int32)
    rad=tri_radiance_mask.to(torch.uint8)
    mod=_module(); n=tri_p.shape[0]; threads=128; blocks=(n+threads-1)//threads
    if blocks:
        shape_args=(np.int32(n),np.int32(height),np.int32(width))
        _launch(mod.get_function("tri_depth"),blocks,threads,(_cp(tri_p),_cp(tri_tau),_cp(rad),_cp(surface_ids),*shape_args,_cp(depth),_cp(first_key),_cp(touches),_cp(depth_only_touches)),tri_p.device)
        _launch(mod.get_function("tri_color"),blocks,threads,(_cp(tri_p),_cp(tri_tau),_cp(tri_color),_cp(rad),_cp(surface_ids),*shape_args,np.float32(depth_eps),_cp(depth),_cp(first_key),_cp(wsum),_cp(rgb)),tri_p.device)
        n_pixels=height*width
        _launch(mod.get_function("extract_id"),(n_pixels+threads-1)//threads,threads,(_cp(first_key),np.int32(n_pixels),_cp(first_id)),tri_p.device)
    env=torch.tensor(environment_rgb,device=tri_p.device,dtype=torch.float32)
    img=env.expand(height*width,3).clone(); finite=torch.isfinite(depth); has=wsum>1e-8
    img[finite]=0.0; img[has]=rgb[has]/wsum[has,None]
    img=img.reshape(height,width,3); depth=depth.reshape(height,width)
    first_id=first_id.reshape(height,width)
    if mirror_mask is not None:
        img=torch.where(mirror_mask[...,None],img,torch.zeros_like(img))
        depth=torch.where(mirror_mask,depth,torch.full_like(depth,float("inf")))
        first_id=torch.where(mirror_mask,first_id,-1)
    return img, depth, first_id, {"pixel_touches": int(touches.item()), "depth_only_pixel_touches": int(depth_only_touches.item())}
