"""Device <-> host transfer options on one logical core (trnblas#61).

Usage: bench_host_link.py [sizes_mb=64,256,1024] [reps=5]
Run pinned to the host CPUs of one core, with NEURON_RT_VISIBLE_CORES set.
"""

import mmap
import statistics
import sys
import time

import torch
import torch_neuronx  # noqa: F401

DEV = "neuron:0"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SIZES_MB = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "64,256,1024").split(",")]


def sync():
    torch_neuronx.synchronize() if hasattr(
        torch_neuronx, "synchronize"
    ) else torch_neuronx.neuron.synchronize()


def thp_empty(n, dtype=torch.float32):
    nbytes = n * torch.empty((), dtype=dtype).element_size()
    m = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    m.madvise(mmap.MADV_HUGEPAGE)
    t = torch.frombuffer(m, dtype=dtype, count=n)
    t._mm = m  # keep alive (attribute on tensor python object)
    return t


def timeit(fn, reps=REPS):
    fn()
    sync()
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        r = fn()
        sync()
        ts.append(time.perf_counter() - t0)
        del r
    return statistics.median(ts)


def run_size(mb):
    n = mb * 1024 * 1024 // 4
    nbytes = n * 4
    h = torch.randn(n)
    d = h.to(DEV)
    sync()
    reuse = torch.empty(n)
    reuse.zero_()
    try:
        pin_reuse = torch.empty(n, pin_memory=True)
        pin_ok = pin_reuse.is_pinned()
    except Exception as e:
        pin_ok = False
        print("pin failed:", e)
    hp = pin_reuse.copy_(h) if pin_ok else None
    cases = {
        "D2H .cpu()": lambda: d.cpu(),
        "D2H .to('cpu')": lambda: d.to("cpu"),
        "D2H empty+copy_": lambda: torch.empty(n).copy_(d),
        "D2H empty+zero_+copy_": lambda: torch.empty(n).zero_().copy_(d),
        "D2H THP-mmap+copy_": lambda: thp_empty(n).copy_(d),
        "D2H reused pageable copy_": lambda: reuse.copy_(d),
    }
    if pin_ok:
        cases["D2H empty(pin)+copy_"] = lambda: torch.empty(n, pin_memory=True).copy_(d)
        cases["D2H reused pinned copy_"] = lambda: pin_reuse.copy_(d)
        cases["D2H reused pinned + clone"] = lambda: pin_reuse.copy_(d).clone()
    cases["host empty+zero_ only"] = lambda: torch.empty(n).zero_()
    cases["host THP-mmap+zero_ only"] = lambda: thp_empty(n).zero_()
    cases["host clone (fresh)"] = lambda: h.clone()
    cases["H2D .to(dev) pageable"] = lambda: h.to(DEV)
    if pin_ok:
        cases["H2D .to(dev) pinned"] = lambda: hp.to(DEV)
        cases["H2D .to(dev) pinned nb"] = lambda: hp.to(DEV, non_blocking=True)
    dreuse = torch.empty(n, device=DEV)
    cases["H2D copy_ into reused dev"] = lambda: dreuse.copy_(h)
    for name, fn in cases.items():
        try:
            t = timeit(fn)
            print(
                f"{mb:6d} MB  {name:30s} {t * 1e3:9.2f} ms  {nbytes / t / 1e9:6.2f} GB/s",
                flush=True,
            )
        except Exception as e:
            print(f"{mb:6d} MB  {name:30s} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)


for mb in SIZES_MB:
    run_size(mb)
