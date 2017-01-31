# Tensor fill benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')
initial_shape = torch.LongStorage({10, 10, 10, 10, 10, 10, 10});

sz = 10000000;
input1 = torch.randn(sz):resize(initial_shape);
input1 = input1:narrow(4,2,2)
input1 = input1:narrow(6,2,2)
input2 = input1:clone()
print("noncontiguous float fill:")
tm = torch.Timer()
for i=1,100 do
   input2 = input1:fill(0)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

input1 = torch.randn(sz):resize(initial_shape);
input1 = input1:narrow(4,2,2)
input1 = input1:narrow(6,2,2)
input2 = input1:clone()
print("noncontiguous double fill:")
tm = torch.Timer()
for i=1,100 do
   input2 = input1:fill(0)
end
print(tm:time().real)

