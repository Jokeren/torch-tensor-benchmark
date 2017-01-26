# Tensor copy benchmark
require("torch")

torch.setdefaulttensortype('torch.FloatTensor')

sz_single = 10000000
input1single = torch.randn(sz_single)
input2single = torch.randn(sz_single)
print("contiguous float copy single thread:")
tm = torch.Timer()
for i=1,1000 do
   input1single:copy(input2single);
end
print(tm:time().real)
