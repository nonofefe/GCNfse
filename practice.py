# import torch

# A = torch.eye(10,10)
# A[:,3] = float('nan')
# print(A)
# B = torch.isnan(A)
# print(B)
# # out, cnt = torch.unique(B, axis=0,return_counts=True)
# # print(out)
# # print(cnt)

# print(torch.__version__)
# C = torch.count_nonzero(C)
# print(C)


import numpy as np

A = np.eye(3,3)
A[:,0] = float('nan')
print(A)
B = np.isnan(A)
print(B)
print(np.count_nonzero(B == False, axis=1))
