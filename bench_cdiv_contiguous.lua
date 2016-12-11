# Tensor cdiv benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz_single = 100000
input1single = torch.randn(sz_single)
input2single = torch.randn(sz_single)
input3single = torch.randn(sz_single)
print("contiguous float div single thread:")
tm = torch.Timer()
for i=1,100 do
   input3single:cdiv(input1single, input2single)
end
print(tm:time().real)

--sz_single * OMP_NUM_THREADS
sz_multi = sz_single * 24
input1multi = torch.randn(sz_multi)
input2multi = torch.randn(sz_multi)
input3multi = torch.randn(sz_multi)
print("contiguous float div multi-thread:")
tm = torch.Timer()
for i=1,100 do
   input3multi:cdiv(input1multi, input2multi)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

sz_single = 100000
input1single = torch.randn(sz_single)
input2single = torch.randn(sz_single)
input3single = torch.randn(sz_single)
print("contiguous double div single thread:")
tm = torch.Timer()
for i=1,100 do
   input3single:cdiv(input1single, input2single)
end
print(tm:time().real)

--sz_single * OMP_NUM_THREADS
sz_multi = sz_single * 24
input1multi = torch.randn(sz_multi)
input2multi = torch.randn(sz_multi)
input3multi = torch.randn(sz_multi)
print("contiguous double div multi-thread:")
tm = torch.Timer()
for i=1,100 do
   input3multi:cdiv(input1multi, input2multi)
end
print(tm:time().real)
