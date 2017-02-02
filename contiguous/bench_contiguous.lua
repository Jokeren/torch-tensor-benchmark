# Tensor contiguous benchmark
require("torch")

local sz = 10000000

function test_fill(input1)
   for i=1,100 do
      input1:fill(0)
   end
end

function test_add(input1, input2)
   for i=1,100 do
      input2:add(input1, 1.0)
   end
end

function test_mul(input1, input2)
   for i=1,100 do
      input2:mul(input1, 1.0)
   end
end

function test_div(input1, input2)
   for i=1,100 do
      input2:div(input1, 1.0)
   end
end

function test_copy(input1, input2)
   for i=1,100 do
      input2:copy(input1)
   end
end

function test_cadd(input1, input2, input3)
   for i=1,100 do
      input3:add(input1, input2)
   end
end

function test_cmul(input1, input2, input3)
   for i=1,100 do
      input3:cmul(input1, input2)
   end
end

function test_cdiv(input1, input2, input3)
   for i=1,100 do
      input3:cdiv(input1, input2)
   end
end

function test_apply_cases(function_call)
   input1 = torch.randn(sz)
   tm = torch.Timer()
   function_call(input1)
   print(tm:time().real)
end

function test_apply2_cases(function_call)
   input1 = torch.randn(sz)
   input2 = input1:clone()
   tm = torch.Timer()
   function_call(input1, input2)
   print(tm:time().real)
end

function test_apply3_cases(function_call)
   input1 = torch.randn(sz)
   input2 = input1:clone()
   input3 = input1:clone()
   tm = torch.Timer()
   function_call(input1, input2, input3)
   print(tm:time().real)
end

function tester()
   local test_apply_functions = {
      ["fill"] = test_fill
   }

   for name, func in pairs(test_apply_functions) do
      print(name)
      test_apply_cases(func)
   end

   local test_apply2_functions = {
      ["add"] = test_add,
      ["mul"] = test_mul,
      ["div"] = test_div,
      ["copy"] = test_copy
   }

   for name, func in pairs(test_apply2_functions) do
      print(name)
      test_apply2_cases(func)
   end

   local test_apply3_functions = {
      ["cadd"] = test_cadd,
      ["cmul"] = test_cmul,
      ["cdiv"] = test_cdiv
   }

   for name, func in pairs(test_apply3_functions) do
      print(name)
      test_apply3_cases(func)
   end
end

torch.setdefaulttensortype('torch.FloatTensor')
print("Float tensor tests")
tester()

torch.setdefaulttensortype('torch.DoubleTensor')
print("Double tensor tests")
tester()
