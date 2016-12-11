# Tensor cmul benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz_single = 100000
input1single = torch.randn(sz_single)
input2single = torch.randn(sz_single)
print("contiguous float mul single thread:")
tm = torch.Timer()
for i=1,100 do
   input1single = input1single * input2single
end
print(tm:time().real)

--sz_single * OMP_NUM_THREADS
sz_multi = sz_single * 24
input1multi = torch.randn(sz_multi)
input2multi = torch.randn(sz_multi)
print("contiguous float mul multi-thread:")
tm = torch.Timer()
for i=1,100 do
   input1multi = input1multi * input2multi
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

sz_single = 100000
input1single = torch.randn(sz_single)
input2single = torch.randn(sz_single)
print("contiguous double mul single thread:")
tm = torch.Timer()
for i=1,100 do
   input1single = input1single * input2single
end
print(tm:time().real)

--sz_single * OMP_NUM_THREADS
sz_multi = sz_single * 24
input1multi = torch.randn(sz_multi)
input2multi = torch.randn(sz_multi)
print("contiguous double mul multi-thread:")
tm = torch.Timer()
for i=1,100 do
   input1multi = input1multi * input2multi
end
print(tm:time().real)

