# Tensor add benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

-- sz > LLC
sz = 10000000
input1 = torch.randn(sz)
print("contiguous float add:")
tm = torch.Timer()
for i=1,100 do
   input1:add(1.0)
end
print(tm:time().real)

torch.setdefaulttensortype('torch.DoubleTensor')

input1 = torch.randn(sz)
print("contiguous double add:")
tm = torch.Timer()
for i=1,100 do
   input1:add(1.0)
end
print(tm:time().real)
