# Tensor cdiv benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz = 10000000
input1 = torch.randn(sz):resize(10000, 1000):transpose(1,2)
input2 = torch.randn(sz):resize(10000, 1000):transpose(1,2)
input3 = torch.randn(sz):resize(10000, 1000):transpose(1,2)
print("transposed float cdiv:")
tm = torch.Timer()
for i=1,100 do
   input3:cdiv(input1, input2)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

input1 = torch.randn(sz):resize(10000, 1000):transpose(1,2)
input2 = torch.randn(sz):resize(10000, 1000):transpose(1,2)
input3 = torch.randn(sz):resize(10000, 1000):transpose(1,2)
print("transposed double cdiv:")
tm = torch.Timer()
for i=1,100 do
   input3:cdiv(input1, input2)
end
print(tm:time().real)