# Torch Contiguous Tensor Benchmark
This benchmark shows the performance of contiguous operations with multi-threading. We use 12 threads with a single processor, avoiding data transfer across different nodes. In each case, we test vectors of 1e^7 elements that is greater than the size of LLC. The benchmark results shows little difference between the `SSE` and `AVX` vectorizations because the performance is bounded by memory bandwidth with low arithmetic intensity.

###Float
Function | SSE (s) | AVX (s) |
-------- | --- | --- |
add |0.22 | 0.25
mul |0.24 | 0.25
div |0.24 | 0.25
cadd |0.48 | 0.47
cmul |0.48 | 0.44
cdiv |0.48 | 0.44
fill |0.39 | 0.02
copy |0.72 | 0.65

The significant benefits of `fill` operation comes from enabling multi-threading.
