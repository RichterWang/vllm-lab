并发曲线： 并发 1、2、4、8；观察吞吐是否继续增加，以及 ITL、TTFT 尾延迟何时开始恶化。
Prefill 压力： 输入长度 128、512、2048，输出长度固定 128；观察 TTFT 和服务端 prefill 时间。
Decode 压力： 输入长度固定 128，输出长度 128、512、1024；观察 TPOT、ITL 和输出吞吐。
排队压力： 在并发测试之外，逐步提高请求到达速率；观察服务端 waiting 与 queue 时间何时明显上升。
缓存压力： 比较重复前缀与不同前缀，并观察 KV cache 使用率和抢占数。

c4_128in_128out
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Maximum request concurrency:             4         
Benchmark duration (s):                  26.23     
Total input tokens:                      12800     
Total generated tokens:                  12800     
Request throughput (req/s):              3.81      
Output token throughput (tok/s):         487.94    
Peak output token throughput (tok/s):    496.00    
Peak concurrent requests:                8.00      
Total token throughput (tok/s):          975.87    
---------------Time to First Token----------------
Mean TTFT (ms):                          31.61     
Median TTFT (ms):                        33.80     
P50 TTFT (ms):                           33.80     
P95 TTFT (ms):                           35.83     
P99 TTFT (ms):                           43.35     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.01      
Median TPOT (ms):                        8.01      
P50 TPOT (ms):                           8.01      
P95 TPOT (ms):                           8.12      
P99 TPOT (ms):                           8.13      
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.01      
Median ITL (ms):                         8.03      
P50 ITL (ms):                            8.03      
P95 ITL (ms):                            8.31      
P99 ITL (ms):                            8.65      
----------------End-to-end Latency----------------
Mean E2EL (ms):                          1049.10   
Median E2EL (ms):                        1050.83   
P50 E2EL (ms):                           1050.83   
P95 E2EL (ms):                           1055.39   
P99 E2EL (ms):                           1057.46   
==================================================