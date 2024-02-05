import torch
state_dict = torch.load('results/temp2.pth', map_location="cpu")
torch.save(state_dict, 'results/old_temp.pth', _use_new_zipfile_serialization=False)
