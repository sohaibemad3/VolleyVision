
"""Batch collation, player padding, and masking."""

import torch


def collate_team_batch(batch, max_players=12):
    """
    Combine samples containing a variable number of players.

    Supports:
        Baseline 3:
            Each sample = (players, group_label)
            players: [N, C, H, W]

        Baseline 6:
            Each sample = (players, player_mask, group_label)
            players: [N, T, C, H, W]
            player_mask: [N]

    Args:
        batch: List of samples returned by DatasetBaseline.
        max_players: Maximum number of player slots in each sample.

    Returns:
        padded_players:
            Baseline 3: [B, max_players, C, H, W]
            Baseline 6: [B, max_players, T, C, H, W]

        player_mask: [B, max_players]
            True  -> a valid player slot.
            False -> a padded or invalid player slot.

        labels: [B], dtype=torch.long
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    if max_players <= 0:
        raise ValueError("max_players must be positive.")

    # Baseline 3 returns two values; baseline 6 returns three.
    sample_size = len(batch[0])

    if sample_size not in (2, 3):
        raise ValueError(
            "Expected (players, label) or "
            "(players, player_mask, label)."
        )

    if any(len(sample) != sample_size for sample in batch):
        raise ValueError("Samples in the batch have inconsistent formats.")

    player_tensors = []
    sample_masks = []
    labels = []

    for sample in batch:
        if sample_size == 2:
            players, label = sample
            original_mask = None
        else:
            players, original_mask, label = sample

        # players must have a player dimension:
        # [N, C, H, W] or [N, T, C, H, W].
        if players.ndim not in (4, 5):
            raise ValueError(
                f"Unexpected player tensor shape: {tuple(players.shape)}"
            )

        num_players = players.shape[0]

        if num_players == 0:
            raise ValueError(
                "A sample has zero player slots; max pooling "
                "over an empty player set is invalid."
            )

        # Build the mask before truncating, preserving the dataset's
        # per-player validity information for baseline 6.
        if original_mask is None:
            mask = torch.ones(num_players, dtype=torch.bool)
        else:
            mask = torch.as_tensor(
                original_mask,
                dtype=torch.bool,
            )

            if mask.ndim != 1 or len(mask) != num_players:
                raise ValueError(
                    "player_mask must have shape [N], matching players."
                )

        # Keep at most max_players. The same truncation is applied
        # to the tensor and its mask so that they remain aligned.
        keep = min(num_players, max_players)
        players = players[:keep]
        mask = mask[:keep]

        # Pad along the player dimension only.
        # Zeros are placeholders, not real player observations.
        if keep < max_players:
            padding_shape = (
                max_players - keep,
                *players.shape[1:],
            )
            padding = players.new_zeros(padding_shape)

            players = torch.cat(
                [players, padding],
                dim=0,
            )

            mask_padding = torch.zeros(
                max_players - keep,
                dtype=torch.bool,
            )
            mask = torch.cat(
                [mask, mask_padding],
                dim=0,
            )

        player_tensors.append(players)
        sample_masks.append(mask)
        labels.append(int(label))

    # Every sample now has the same number of player slots.
    padded_players = torch.stack(player_tensors, dim=0)
    player_mask = torch.stack(sample_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_players, player_mask, labels
