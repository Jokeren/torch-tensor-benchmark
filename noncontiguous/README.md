# Torch Noncontiguous tensor Benchmarks

Multi-threading is not supported for TH_TENSOR_APPLY

###Float
Function | Read One Noncontiguous (s) | Optimized (s) | Read Both Noncontiguous (s) | Optimized (s) | Store Noncontiguous (s) | Optimized (s) | All Noncontiguous (s) | Optimized (s)
-------- | --- | --- | --- | --- | --- | --- | --- | --- |
add |0.26 | **0.24** | | | 0.11 | **0.04** | 0.27 | **0.25**
cadd |0.43| **0.42** |0.49|**0.48** |0.42| 0.42 | 0.56 | **0.55**
mul |0.27 | **0.24** | | | 0.12 | **0.04** | 0.27 | **0.25** 
cmul |0.43| **0.42** |0.49|**0.48** |0.44|**0.41**|0.56| **0.55**
div |0.56 | 0.56     | | |0.55  | **0.15** | 0.56 | 0.56
cdiv |0.59| 0.60     |0.62|0.62     |0.58| 0.58   |0.66| **0.65**
fill |0.15| 0.15     | | |0.04  | 0.04     | 0.15 | 0.15
copy |0.27| **0.26** | | |0.27  | 0.27     | 0.40 | 0.40
