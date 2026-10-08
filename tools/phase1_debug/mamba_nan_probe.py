"""Diagnostic: find the first non-finite value in the Mamba-3 complex SSD.

Run: python3 tools/phase1_debug/mamba_nan_probe.py
"""
import torch
import torch.nn.functional as F

from models.ssd_complex import (
    ssd_complex_chunkwise, ssd_naive_complex, per_chunk_ssd_pytorch, _discretise,
)


def report(name, t):
    if t.is_complex():
        finite = torch.isfinite(t.real) & torch.isfinite(t.imag)
    else:
        finite = torch.isfinite(t)
    ok = bool(finite.all())
    mx = t.abs().max().item()
    print(f"  {name:28s} dtype={str(t.dtype):16s} shape={tuple(t.shape)} finite={ok} max|.|={mx:.3e}")
    return ok


def probe(tag, dtype, autocast):
    torch.manual_seed(0)
    B, T, H, D, N = 2, 128, 2, 4, 4
    dev = "cpu"
    x = torch.randn(B, T, H, D, device=dev)
    A = (torch.randn(H, device=dev, dtype=torch.complex64) - 1.0)
    B_r = torch.randn(B, T, H, N, device=dev)
    C_r = torch.randn(B, T, H, N, device=dev)
    dt = torch.randn(B, T, H, device=dev)

    if dtype is not None:
        x = x.to(dtype); B_r = B_r.to(dtype); C_r = C_r.to(dtype); dt = dt.to(dtype)

    print(f"\n=== {tag} (input dtype={dtype}, autocast={autocast}) ===")
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        B_t = torch.complex(B_r, torch.randn_like(B_r))
        C_t = torch.complex(C_r, torch.randn_like(C_r))
        A_log = F.softplus(dt) * A
        report("softplus(dt)", F.softplus(dt))
        report("A_log = softplus*A", A_log)

        xc = x.float()
        report("x (as fp32)", xc)

        # replicate internal chunking
        Cc_size = 64
        pad = (Cc_size - (T % Cc_size)) % Cc_size
        xp = F.pad(xc, (0, 0, 0, 0, 0, pad)) if pad else xc
        Bp = F.pad(B_t, (0, 0, 0, 0, 0, pad)) if pad else B_t
        Cp = F.pad(C_t, (0, 0, 0, 0, 0, pad)) if pad else C_t
        Ap = F.pad(A_log, (0, 0, 0, pad)) if pad else A_log
        Tp = T + pad
        nch = Tp // Cc_size

        def ch(t): return t.reshape(B, nch, Cc_size, *t.shape[2:])
        Xc, Bc, Cc, Ac = ch(xp).to(torch.complex64), ch(Bp), ch(Cp), ch(Ap)
        report("Xc", Xc); report("Bc", Bc); report("Cc", Cc); report("Ac", Ac)

        A_cumsum = torch.cumsum(Ac, dim=2)
        report("A_cumsum", A_cumsum)
        report("A_cumsum real min/max", A_cumsum.real.aminmax()[0].unsqueeze(0))
        decay_states = torch.exp(A_cumsum[:, :, -1:, :] - A_cumsum)
        report("decay_states", decay_states)

        Y_diag, states = per_chunk_ssd_pytorch(Bc, Cc, Xc, Ac, decay_states)
        report("Y_diag", Y_diag); report("states", states)

        chunk_decay = A_cumsum[:, :, -1, :]
        cd = chunk_decay.permute(0, 2, 1).contiguous()
        cd_cs = torch.cumsum(cd, dim=-1)
        cd_shift = torch.cat([torch.zeros_like(cd_cs[..., :1]), cd_cs[..., :-1]], dim=-1)
        cd_seg = cd_shift.unsqueeze(-1) - cd_cs.unsqueeze(-2)
        report("cd_seg", cd_seg)
        decay_chunk = torch.exp(cd_seg) * torch.tril(
            torch.ones(nch, nch, device=dev, dtype=torch.bool), diagonal=-1)
        report("decay_chunk(to states.dtype)", decay_chunk.to(states.dtype))
        st = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk.to(states.dtype), states)
        report("states after cross-chunk", st)
        report("exp(A_cumsum)", torch.exp(A_cumsum))
        Y_off = torch.einsum("bclhn,bchpn,bclh->bclhp", Cc.to(st.dtype), st,
                             torch.exp(A_cumsum).to(st.dtype))
        report("Y_off", Y_off)
        Y = (Y_diag + Y_off).real
        report("Y final", Y)


if __name__ == "__main__":
    probe("fp32 eager", torch.float32, False)
    probe("bf16-cast inputs, no autocast", torch.bfloat16, False)
    probe("fp32 inputs under bf16 autocast", torch.float32, True)
