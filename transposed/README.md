# Torch Transposed Tensor Benchmark 
Multi-threading is not supported for `TH_TENSOR_APPLY`s. In the transposed benchmark, we set up large vectors (1e^7) and transpose the lowest dimension. In this way, we test the effects of optimzing stride accesses. Particularly, we directly apply contiguous operations for `TH_TENSOR_APPLY` with `fill`. And instead of accessing contiguous elements in each iteration, we access the last contiguous section without updating dimension counters. Therefore, with the reduction of extra operations, we achieve at most three four times speedup.

###Float
*TH_TENSOR_APPLY*

Function | Transposed (s) | Optimized (s) |
-------- | ---- | ---- |
fill     | 21.3 | 0.02 |

*TH_TENSOR_APPLY2*

Function | Read Transposed (s) | Optimized (s) | Store Transposed (s) | Optimized (s) | All Transposed (s) | Optimized (s)
-------- | --- | --- | --- | --- | --- | --- |
add      | 8.8 | **3.2** | 21.9 | **4.4** | 22.2 | **6.2**
mul      | 8.8 | **3.2** | 21.9 | **4.4** | 22.2 | **6.2** 
div      | 8.8 | **3.2** | 21.9 | **4.2** | 22.2 | **5.8**
copy     | 8.8 | **3.2** | 21.9 | **4.2** | 22.2 | **5.8**

*TH_TENSOR_APPLY3*

Function | Read One Transposed (s) | Optimized (s) | Read Both Transposed (s) | Optimized (s) | Store Transposed (s) | Optimized (s) | All Transposed (s) | Optimized (s)
-------- | --- | --- | --- | --- | --- | --- | --- | --- |
cadd     | 11.5 | **4.6** | 14.4 | **5.7** | 22.1 | **5.2** | 24.1 | **9.2**
cmul     | 11.5 | **4.6** | 15.1 | **5.9** | 22.1 | **5.2** | 23.8 | **9.2**
cdiv     | 11.5 | **4.6** | 14.4 | **6.1** | 22.1 | **5.2** | 23.8 | **9.2**
