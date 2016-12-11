# Tensor multi-thread correctness
require("torch")
sz_single = 100000

torch.setdefaulttensortype('torch.FloatTensor')

--sz_single * OMP_NUM_THREADS
sz_multi = sz_single * 24
input1multi = torch.randn(sz_multi)
input2multi = input1multi:clone()
print("contiguous float add multi-thread:")
tm = torch.Timer()
input1multi:add(1.0)
print(tm:time().real)
for j=1,sz_multi do
   input2multi[j] = input2multi[j] + 1.0
end
local err = (input2multi-input1multi):abs():max()
print("error:", err)

torch.setdefaulttensortype('torch.DoubleTensor')

--sz_single * OMP_NUM_THREADS
sz_multi = sz_single * 24
input1multi = torch.randn(sz_multi)
input2multi = input1multi:clone()
print("contiguous double add multi-thread:")
tm = torch.Timer()
input1multi:add(1.0)
print(tm:time().real)
for j=1,sz_multi do
   input2multi[j] = input2multi[j] + 1.0
end
local err = (input2multi-input1multi):abs():max()
print("error:", err)
