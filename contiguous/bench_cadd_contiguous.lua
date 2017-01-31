# Tensor cadd benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz = 10000000
input1 = torch.randn(sz)
input2 = torch.randn(sz)
input3 = torch.randn(sz)
print("contiguous float cadd:")
tm = torch.Timer()
for i=1,100 do
   input3:add(input1, input2)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

input1 = torch.randn(sz)
input2 = torch.randn(sz)
input3 = torch.randn(sz)
print("contiguous double cadd:")
tm = torch.Timer()
for i=1,100 do
   input3:add(input1, input2)
end
print(tm:time().real)
