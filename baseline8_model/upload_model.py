import torch
def upload_model(path):
	checkpoint = torch.load(
	    path,
	    map_location=device,
	    weights_only=True
	)

	model.load_state_dict(checkpoint)
	return model

