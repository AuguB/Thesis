from flows import *


dim = 5
actnorm = actnorm(5)

tries= 10
with torch.no_grad():
    for i in range(tries):
        rand = torch.rand((2,dim))
        forward, det = actnorm(rand)
        backward = actnorm.inverse(forward)
        print(backward-rand)
