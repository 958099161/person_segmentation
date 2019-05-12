import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):
	def __init__(self, weight=None):
		super().__init__()
		self.loss = nn.NLLLoss2d(weight)
		self.loss = nn.CrossEntropyLoss(size_average=False, reduce=False)
		#self.loss=nn.MSEloss()
	def forward(self, outputs, targets):
        	#torch version >0.2 F.log_softmax(input, dim=?) 
        	#dim (int): A dimension along which log_softmax will be computed.
		# try:

		# out = F.log_softmax(outputs, dim=1)
		out_loss= self.loss(outputs, targets.long())

		mask = targets >= 0

		out_l = torch.sum(torch.masked_select(out_loss, mask)) / torch.sum(mask.float())
		return out_l

		# except TypeError as t:
		# 	return self.loss(F.log_softmax(outputs), targets)       #else

