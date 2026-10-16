import sys, time, statistics
from pathlib import Path
import torch
sys.path.insert(0, '/workspace/vllm_lab/worktrees/vllm-60097-candidate')
torch.empty(1, device='cuda')
torch.cuda.synchronize()
import vllm
vllm.__path__.append('/workspace/vllm_lab/src/vllm/vllm')
from vllm.v1.worker.gpu.sample import logprob_autotune as a
cache=Path('/workspace/vllm_lab_h100/results/kernel/issue60095/autotune/cache')
original=a._fingerprint
def timed_fingerprint(*args):
    t=time.perf_counter(); result=original(*args); dt=(time.perf_counter()-t)*1000
    print('fingerprint_ms',round(dt,3),flush=True)
    return result
a._fingerprint=timed_fingerprint
samples=[]
for n in range(30):
    t=time.perf_counter()
    result=a.initialize_split_autotune(151936,torch.bfloat16,6,'cuda',cache)
    dt=(time.perf_counter()-t)*1000
    samples.append(dt)
    print('hit',n,result['status'],round(dt,3),flush=True)
print('summary_ms',min(samples),statistics.median(samples),max(samples),flush=True)
