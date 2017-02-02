# Torch Noncontiguous Tensor Benchmark
Multi-threading is not supported for `TH_TENSOR_APPLY`s. In the noncontiguous benchmark, we test the benefits of dimension collapsing. Hence, we use small size vectors (2^16) in each test case. The optimized implementations outperform the origin ones with up to 100%, as the executions of dimension counters are reduced.

###Float
*TH_TENSOR_APPLY*

Function | Noncontiguous (s) | Optimized (s) |
-------- | ---- | ---- |
fill     | 0.012 | 0.003 |

*TH_TENSOR_APPLY2*

Function | Read Noncontiguous (s) | Optimized (s) | Store Noncontiguous (s) | Optimized (s) | All Noncontiguous (s) | Optimized (s)
-------- | --- | --- | --- | --- | --- | --- |
add      |0.010 | **0.05** | 0.010 | **0.05** | 0.014 | **0.08**
mul      |0.010 | **0.05** | 0.010 | **0.05** | 0.014 | **0.08** 
div      |0.009 | **0.05** | 0.009 | **0.07** | 0.014 | **0.08**
copy     |0.009 | **0.05** | 0.010 | **0.05** | 0.014 | **0.08**

*TH_TENSOR_APPLY3*

Function | Read One Noncontiguous (s) | Optimized (s) | Read Both Noncontiguous (s) | Optimized (s) | Store Noncontiguous (s) | Optimized (s) | All Noncontiguous (s) | Optimized (s)
-------- | --- | --- | --- | --- | --- | --- | --- | --- |
cadd     | 0.010 | **0.007** | 0.015 | **0.010** | 0.010 | **0.007** | 0.020 | **0.013**
cmul     | 0.010 | **0.007** | 0.015 | **0.010** | 0.010 | **0.007** | 0.020 | **0.013**
cdiv     | 0.010 | **0.007** | 0.015 | **0.010** | 0.010 | **0.007** | 0.020 | **0.014**
