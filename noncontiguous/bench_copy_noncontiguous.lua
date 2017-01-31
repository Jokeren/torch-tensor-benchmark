# Tensor copy benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')
initial_shape = torch.LongStorage({10, 10, 10, 10, 10, 10, 10})

sz = 10000000
input1 = torch.randn(sz):resize(initial_shape)
input1 = input1:narrow(7,2,2)
input2 = torch.randn(sz):resize(initial_shape)
input2 = input2:narrow(7,2,2)
print("noncontiguous float copy:")
tm = torch.Timer()
for i=1,100 do
   input1:copy(input2);
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

input1 = torch.randn(sz):resize(initial_shape)
input1 = input1:narrow(7,2,2)
input2 = torch.randn(sz):resize(initial_shape)
input2 = input2:narrow(7,2,2)
print("noncontiguous double copy:")
tm = torch.Timer()
for i=1,100 do
   input1:copy(input2);
end
print(tm:time().real)
