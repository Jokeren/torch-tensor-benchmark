# Torch Tensor Benchmarks
##Environment
- E5-2680 v3 @ 2.50GHz (AVX2, FMA)
- 12 threads
- 1e^7 elements

##Single Thread

###Float
Function | SSE | AVX 
-------- | --- | --- 
add | 0.0059 | 0.0031 
cadd | 0.0081 | 0.0069
div | 0.021 | 0.0058
cdiv | 0.008| 0.0069
mul |0.0059 | 0.0031
cmul | 0.008| 0.007
fill |0.024 | 0.024

###Double
Function | SSE | AVX
-------- | --- | ---
add | 0.0065 | 0.0047
cadd | 0.0092 | 0.0081
div | 0.0025 | 0.0012
cdiv | 0.021 | 0.021
mul |0.0073 | 0.0044
cmul |0.001 | 0.0081
fill |0.038 | 0.036

##Multi-thread

###Float
Function | SSE | AVX 
-------- | --- | --- 
add | 0.0080 | 0.0037
cadd | 0.0074| 0.0057
div | 0.025 | 0.0073
cdiv | 0.0077 | 0.0074
mul | 0.0074| 0.0036
cmul | 0.0070 | 0.0056
fill |

###Double
Function | SSE | AVX 
-------- | --- | --- 
add | 0.0079 | 0.051 
cadd | 0.012 | 0.1
div | 0.028 | 0.012
cdiv | 0.025 | 0.025
mul | 0.008 | 0.0053
cmul | 0.012 | 0.01
fill |
