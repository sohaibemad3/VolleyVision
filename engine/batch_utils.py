def unpack_batch(batch):
    """
    Convert a DataLoader batch into the canonical order:
        inputs, labels, player_mask

    Supported formats:
        (inputs, labels)
        (inputs, player_mask, labels)  # current volleyball collate function
    """
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Expected tuple/list batch, got {type(batch).__name__}.")

    if len(batch) == 2:
        inputs, labels = batch
        player_mask = None
    elif len(batch) == 3:
        inputs, player_mask, labels = batch
    else:
        raise ValueError(f"Expected 2 or 3 batch elements, got {len(batch)}.")

    return inputs, labels, player_mask
