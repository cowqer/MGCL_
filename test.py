import torch, time
x = torch.randn(10000, 10000, device="cuda")
torch.cuda.synchronize()
t1 = time.time()
for i in range(1000):
    y = x @ x
torch.cuda.synchronize()
print("Time:", time.time() - t1)

