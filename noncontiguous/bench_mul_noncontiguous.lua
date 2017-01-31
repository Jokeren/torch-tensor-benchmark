# Tensor mul benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz = 10000000
input1 = torch.randn(sz):resize(initial_shape)
input1 = input1:narrow(4,2,2)
input1 = input1:narrow(6,2,2)
input2 = input1:clone()
print("noncontiguous float mul:")
tm = torch.Timer()
for i=1,100 do
   input2 = input1:mul(1.0)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

input1 = torch.randn(sz):resize(initial_shape)
input1 = input1:narrow(4,2,2)
input1 = input1:narrow(6,2,2)
input2 = input1:clone()
print("noncontiguous double mul:")
tm = torch.Timer()
for i=1,100 do
   input2 = input1:mul(1.0)
end
print(tm:time().real)
