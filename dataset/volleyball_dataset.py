
"""PyTorch datasets for volleyball group activity recognition."""

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .collate import collate_team_batch


class DatasetBaseline(Dataset):
    """
    Dataset wrapper for the different volleyball baselines.

    Supported baselines:
        1: Full-image group classification.
        2: Cropped-player action classification.
        3: Cropped players from one frame, group classification.
        5: Full-image sequence classification.
        6: Cropped-player sequence, group classification.

    The dataset prepares one sample at a time.
    Variable-player padding is handled by collate.py.
    """

    MAX_PLAYERS = 12

    GROUP_CATEGORIES = {
        "l-pass": 0,
        "r-pass": 1,
        "l-spike": 2,
        "r_spike": 3,
        "l_set": 4,
        "r_set": 5,
        "l_winpoint": 6,
        "r_winpoint": 7,
    }

    PERSON_CATEGORIES = {
        "waiting": 0,
        "setting": 1,
        "digging": 2,
        "falling": 3,
        "spiking": 4,
        "blocking": 5,
        "jumping": 6,
        "moving": 7,
        "standing": 8,
    }

    def __init__(
        self,
        baseline_dataset,
        choice_baseline=1,
        transform=None,
    ):
        super().__init__()

        self.baseline_dataset = baseline_dataset
        self.choice_baseline = choice_baseline
        self.transform = transform

        supported_baselines = {1, 2, 3, 5, 6}
        if choice_baseline not in supported_baselines:
            raise ValueError(
                f"Unsupported baseline: {choice_baseline}. "
                f"Choose one of {sorted(supported_baselines)}."
            )

    def __len__(self):
        """Return the number of samples."""
        return len(self.baseline_dataset)

    @staticmethod
    def _load_rgb_image(image_path):
        """Open an image safely and return an independent RGB image."""
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def _apply_transform(self, image):
        """Apply the selected image transformations, if provided."""
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _group_label(self, label):
        """Convert a group activity label to its integer ID."""
        if label not in self.GROUP_CATEGORIES:
            raise ValueError(f"Unknown group label: {label}")
        return self.GROUP_CATEGORIES[label]

    def _person_label(self, label):
        """Convert an individual player action to its integer ID."""
        if label not in self.PERSON_CATEGORIES:
            raise ValueError(f"Unknown player action: {label}")
        return self.PERSON_CATEGORIES[label]

    def image_classification(self, index):
        """
        Baseline 1:
        Classify group activity using the full image.

        Expected sample:
            (image_path, group_label)

        Returns:
            image: [C, H, W]
            label: integer
        """
        image_path, label = self.baseline_dataset[index]

        image = self._load_rgb_image(image_path)
        image = self._apply_transform(image)

        return image, self._group_label(label)

    def cropped_player_classification(self, index):
        """
        Baseline 2:
        Classify one player's action from a bounding-box crop.

        Expected sample:
            (image_path, box, player_label)

        Returns:
            cropped_image: [C, H, W]
            label: integer
        """
        image_path, box, label = self.baseline_dataset[index]

        x1, y1, x2, y2 = box

        image = self._load_rgb_image(image_path)
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image = self._apply_transform(cropped_image)

        return cropped_image, self._person_label(label)

    def team_of_cropped_player(self, index):
        """
        Baseline 3:
        Classify group activity using all player crops in one frame.

        Expected sample:
            (box_1, box_2, ..., box_N, group_label, image_path)

        Returns:
            players: [N, C, H, W]
            label: integer

        N may differ between samples. Do not pad here.
        """
        sample = self.baseline_dataset[index]

        boxes = sample[:-2]
        group_label = sample[-2]
        image_path = sample[-1]

        image = self._load_rgb_image(image_path)

        player_crops = []

        for box in boxes:
            x1, y1, x2, y2 = box
            crop = image.crop((x1, y1, x2, y2))
            crop = self._apply_transform(crop)
            player_crops.append(crop)

        if not player_crops:
            raise ValueError(
                f"Sample {index} contains no player bounding boxes."
            )

        # All transformed crops must have the same shape to be stacked.
        players = torch.stack(player_crops, dim=0)

        return players, self._group_label(group_label)

    def seq_of_image(self, index):
        """
        Baseline 5:
        Classify group activity using a sequence of full images.

        Expected sample:
            (frame_1_path, ..., frame_T_path, group_label)

        Returns:
            frames: [T, C, H, W]
            label: integer
        """
        sample = self.baseline_dataset[index]

        frame_paths = sample[:-1]
        group_label = sample[-1]

        frames = []

        for image_path in frame_paths:
            image = self._load_rgb_image(image_path)
            image = self._apply_transform(image)
            frames.append(image)

        if not frames:
            raise ValueError(f"Sample {index} contains no frames.")

        frames = torch.stack(frames, dim=0)

        return frames, self._group_label(group_label)

    def get_cropped_player_sequence_with_team(self, index):
        """
        Baseline 6:
        Classify group activity using player crops across multiple frames.

        Expected sample:
            (
                frame_1_info,
                ...,
                frame_T_info,
                group_label,
            )

        Each frame_info contains:
            [image_path, box_1, ..., box_N]

        Each box is:
            (x1, y1, x2, y2, player_label)

        Returns:
            players: [N, T, C, H, W]
            player_mask: [N], True for a player detected in at least
                         one frame and False for a player slot that
                         was padded within this sample.
            group_label: integer

        IMPORTANT:
        This implementation assumes player index i refers to the
        same player across frames. It does not perform player tracking.
        """
        sample = self.baseline_dataset[index]

        frame_infos = sample[:-1]
        group_label = sample[-1]

        if not frame_infos:
            raise ValueError(f"Sample {index} contains no frames.")

        # Store the crops for each frame before stacking.
        # This also allows the number of detections to vary by frame.
        crops_by_frame = []
        max_players_in_sample = 0

        for frame_info in frame_infos:
            image_path = frame_info[0]
            image = self._load_rgb_image(image_path)

            frame_crops = []

            for box_info in frame_info[1:]:
                # The final value is the player's action label.
                x1, y1, x2, y2 = box_info[:4]

                crop = image.crop((x1, y1, x2, y2))
                crop = self._apply_transform(crop)
                frame_crops.append(crop)

            crops_by_frame.append(frame_crops)
            max_players_in_sample = max(
                max_players_in_sample,
                len(frame_crops),
            )

        if max_players_in_sample == 0:
            raise ValueError(
                f"Sample {index} contains no player detections."
            )

        # Infer the transformed image shape to create zero padding.
        example_crop = next(
            crop
            for frame_crops in crops_by_frame
            for crop in frame_crops
        )
        zero_crop = torch.zeros_like(example_crop)

        # Create a [T, N, C, H, W] tensor.
        # Missing detections at a particular frame get zero-filled slots.
        padded_frames = []
        player_seen = torch.zeros(
            max_players_in_sample,
            dtype=torch.bool,
        )

        for frame_crops in crops_by_frame:
            num_detected = len(frame_crops)

            if num_detected:
                frame_tensor = torch.stack(frame_crops, dim=0)
                player_seen[:num_detected] = True
            else:
                frame_tensor = example_crop.new_empty(
                    (0, *example_crop.shape)
                )

            missing_players = max_players_in_sample - num_detected

            if missing_players:
                frame_padding = zero_crop.unsqueeze(0).expand(
                    missing_players, *zero_crop.shape
                )
                frame_tensor = torch.cat(
                    (frame_tensor, frame_padding),
                    dim=0,
                )

            padded_frames.append(frame_tensor)

        # [T, N, C, H, W] -> [N, T, C, H, W]
        players = torch.stack(padded_frames, dim=0)
        players = players.permute(1, 0, 2, 3, 4).contiguous()

        return players, player_seen, self._group_label(group_label)

    def __getitem__(self, index):
        """Dispatch the requested baseline."""
        if self.choice_baseline == 1:
            return self.image_classification(index)

        if self.choice_baseline == 2:
            return self.cropped_player_classification(index)

        if self.choice_baseline == 3:
            return self.team_of_cropped_player(index)

        if self.choice_baseline == 5:
            return self.seq_of_image(index)

        if self.choice_baseline == 6:
            return self.get_cropped_player_sequence_with_team(index)

        # Normally unreachable because __init__ validates the baseline.
        raise RuntimeError("Invalid baseline configuration.")


def load_dataset(
    baseline_dataset,
    choice_baseline=1,
    transform=None,
    shuffle=True,
    batch_size=32,
    pin_memory=True,
    num_workers=4,
    max_players=12,
):
    """
    Create a DataLoader for the selected baseline.

    Baselines 3 and 6 use collate_team_batch to pad variable
    player counts and return a player mask.
    """
    dataset = DatasetBaseline(
        baseline_dataset=baseline_dataset,
        choice_baseline=choice_baseline,
        transform=transform,
    )

    collate_fn = None

    if choice_baseline in (3, 6):
        # Use a partial so DataLoader can pass each batch to the
        # collate function with the configured maximum player count.
        from functools import partial

        collate_fn = partial(
            collate_team_batch,
            max_players=max_players,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
