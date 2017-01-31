# Tensor mul benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz = 10000000
input1 = torch.randn(sz):resize(10000, 1000):transpose(1,2)
print("transposed float mul:")
tm = torch.Timer()
for i=1,100 do
   input1:mul(1.0)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

input1 = torch.randn(sz):resize(10000, 1000):transpose(1,2)
print("transposed double mul:")
tm = torch.Timer()
for i=1,100 do
   input1:mul(1.0)
end
print(tm:time().real)
