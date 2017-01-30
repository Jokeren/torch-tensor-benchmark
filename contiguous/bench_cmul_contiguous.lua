# Tensor cmul benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz_single = 100000
input1single = torch.randn(sz_single)
input2single = torch.randn(sz_single)
input3single = torch.randn(sz_single)
print("contiguous float cmul single thread:")
tm = torch.Timer()
for i=1,100 do
   input3single:cmul(input1single, input2single)
end
print(tm:time().real)

--sz_single * OMP_NUM_THREADS
sz_multi = sz_single * 12
input1multi = torch.randn(sz_multi)
input2multi = torch.randn(sz_multi)
input3multi = torch.randn(sz_multi)
print("contiguous float cmul multi-thread:")
tm = torch.Timer()
for i=1,100 do
   input3multi:cmul(input1multi, input2multi)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

sz_single = 100000
input1single = torch.randn(sz_single)
input2single = torch.randn(sz_single)
input3single = torch.randn(sz_single)
print("contiguous double cmul single thread:")
tm = torch.Timer()
for i=1,100 do
   input3single:cmul(input1single, input2single)
end
print(tm:time().real)

--sz_single * OMP_NUM_THREADS
sz_multi = sz_single * 12
input1multi = torch.randn(sz_multi)
input2multi = torch.randn(sz_multi)
input3multi = torch.randn(sz_multi)
print("contiguous double cmul multi-thread:")
tm = torch.Timer()
for i=1,100 do
   input3multi:cmul(input1multi, input2multi)
end
print(tm:time().real)
