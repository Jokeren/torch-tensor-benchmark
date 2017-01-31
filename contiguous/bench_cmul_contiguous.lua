# Tensor cmul benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz = 10000000
input1 = torch.randn(sz)
input2 = torch.randn(sz)
input3 = torch.randn(sz)
print("contiguous float cmul:")
tm = torch.Timer()
for i=1,100 do
   input3:cmul(input1, input2)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

input1 = torch.randn(sz)
input2 = torch.randn(sz)
input3 = torch.randn(sz)
print("contiguous double cmul:")
tm = torch.Timer()
for i=1,100 do
   input3:cmul(input1, input2)
end
print(tm:time().real)

