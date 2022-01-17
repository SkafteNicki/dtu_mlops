import torch
from torch import ResnetFromTorchVision

model = ResnetFromTorchVision(pretrained=True)
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')

input = torch.rand([1,3,224, 224])
unscripted_top5_indices = torch.topk(model(input),5)
scripted_top5_indices = torch.topk(script_model(input),5)
assert torch.allclose(unscripted_top5_indices, scripted_top5_indices)