# Tensor div benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz = 10000000
input1 = torch.randn(sz)
print("contiguous float div:")
tm = torch.Timer()
for i=1,100 do
   input1:div(1.0)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

input1 = torch.randn(sz)
print("contiguous double div:")
tm = torch.Timer()
for i=1,100 do
   input1:div(1.0)
end
print(tm:time().real)
