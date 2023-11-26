import torch
import torch.nn as nn
import torch.nn.functional as F

# mask with -1e9
# Simulate logits (attention scores before softmax)
logits = torch.tensor([[2.0, 3.0, 1.0, 4.0],
                       [1.0, 2.0, 3.0, 4.0]])

# Mask (1s represent positions to keep, 0s to ignore)
mask = torch.tensor([[1, 1, 0, 0],
                     [1, 0, 0, 0]])

# Apply mask
masked_logits = logits.masked_fill(mask == 0, -1e9)

# Softmax
probabilities = F.softmax(masked_logits, dim=-1)
print(probabilities)

# define target labels
targets = torch.tensor([1, 0]) 
# Define a target distribution (should sum to 1 across dim=-1)
target_prob = torch.tensor([[0, 1, 0.0, 0.0],
                       [1.0, 0.0, 0.0, 0.0]])

# Convert target to log probabilities
# target_log_prob = torch.log(target_prob)

# Define the KL divergence loss function
kl_loss = nn.KLDivLoss(reduction='sum')
# Calculate loss (Note: input should be log probabilities)
loss = kl_loss(F.log_softmax(masked_logits, dim=-1), target_prob)
print("KL Divergence Loss with mask -1e9:", loss)


# mask with np.NINF
import numpy as np

# Reusing the same logits
# Apply mask
masked_logits_ninf = logits.masked_fill(mask == 0, float(np.NINF))

# # Softmax
# probabilities_ninf = F.softmax(masked_logits_ninf, dim=-1)
# # print(probabilities_ninf)
# log_probabilities_ninf = torch.log(probabilities_ninf)
# print(log_probabilities_ninf)
loss2 = kl_loss(F.log_softmax(masked_logits_ninf, dim=-1), target_prob)
print("KL Divergence Loss with mask np.NINF:", loss2)


'''
This function is more numerically stable than using a plain Softmax followed by a Negative Log Likelihood Loss
'''
loss3 = F.cross_entropy(masked_logits_ninf, targets, reduction='sum')
print("cross_entropyLoss with mask np.NINF:", loss3)

