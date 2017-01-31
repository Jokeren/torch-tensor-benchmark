# Tensor fill benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz = 10000000
input1 = torch.randn(sz)
print("contiguous float fill:")
tm = torch.Timer()
for i=1,100 do
   input1:fill(0)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

sz = 10000000
input1 = torch.randn(sz)
print("contiguous double fill:")
tm = torch.Timer()
for i=1,100 do
   input1:fill(0)
end
print(tm:time().real)
