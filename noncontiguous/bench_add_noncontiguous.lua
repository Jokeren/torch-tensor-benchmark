# Tensor add benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')
initial_shape = torch.LongStorage({10, 10, 10, 10, 10, 10, 10})

sz_single = 10000000
input1single = torch.randn(sz_single):resize(initial_shape)
input1single = input1single:narrow(4,2,2)
input1single = input1single:narrow(6,2,2)
input2single = input1single:clone()
print("contiguous float add single thread:")
tm = torch.Timer()
for i=1,100 do
   input2single = input1single:add(1.0)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

sz_single = 10000000
input1single = torch.randn(sz_single):resize(initial_shape)
input1single = input1single:narrow(4,2,2)
input1single = input1single:narrow(6,2,2)
input2single = input1single:clone()
print("contiguous double add single thread:")
tm = torch.Timer()
for i=1,100 do
   input2single = input1single:add(1.0)
end
print(tm:time().real)
