# Tensor cadd benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')
initial_shape = torch.LongStorage({10, 10, 10, 10, 10, 10, 10})

sz_single = 10000000
input1single = torch.randn(sz_single):resize(initial_shape)
input1single = input1single:narrow(4,2,2)
input1single = input1single:narrow(6,2,2)
input2single = input1single:clone()
input3single = input1single:clone()
print("contiguous float cadd single thread:")
tm = torch.Timer()
for i=1,100 do
   input3single:add(input1single, input2single)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

sz_single = 10000000
input1single = torch.randn(sz_single):resize(initial_shape)
input1single = input1single:narrow(4,2,2)
input1single = input1single:narrow(6,2,2)
input2single = input1single:clone()
input3single = input1single:clone()
print("contiguous double cadd single thread:")
tm = torch.Timer()
for i=1,100 do
   input3single:add(input1single, input2single)
end
print(tm:time().real)
