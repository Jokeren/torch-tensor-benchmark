# Tensor fill benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz_single = 1000000
input1single = torch.randn(sz_single)
print("contiguous float fill single thread:")
tm = torch.Timer()
for i=1,100 do
   input1single:fill(0)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

sz_single = 1000000
input1single = torch.randn(sz_single)
print("contiguous double fill single thread:")
tm = torch.Timer()
for i=1,100 do
   input1single:fill(0)
end
print(tm:time().real)