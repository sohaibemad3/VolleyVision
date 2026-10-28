
"""Prepare sample dictionaries for volleyball activity recognition baselines."""

import os


class BaselinePreparation:
    """
    Convert volleyball annotations into sample dictionaries.

    Each method prepares the data format expected by a specific baseline.

    Expected annotation structure:
        videos_annot[video_id][frame_id] = {
            "category": group_activity_label,
            "frame_boxes_dct": {
                frame_id: [player_annotation, ...],
                ...
            }
        }

    Each player annotation is expected to provide:
        player.box       -> (x1, y1, x2, y2)
        player.category  -> individual action label

    Note:
        This class prepares paths, bounding boxes, and labels.
        It does not load image pixels, apply transformations,
        or create PyTorch tensors.
    """

    def __init__(self, videos_root, videos_annot):
        """
        Args:
            videos_root: Root directory containing extracted video frames.
            videos_annot: Parsed annotations indexed by video and frame IDs.
        """
        self.videos_root = videos_root
        self.videos_annot = videos_annot

    def _get_image_path(self, video_id, frame_id):
        """
        Build the path to an image frame.

        Expected directory structure:
            videos_root/
                video_id/
                    annotation_frame_id/
                        image_frame_id.jpg

        The directory name comes from the annotated sample frame,
        while the filename comes from the requested frame ID.
        """
        return os.path.join(
            self.videos_root,
            str(video_id),
            str(frame_id),
            f"{frame_id}.jpg",
        )

    def _get_frame_info(self, video_id, frame_id):
        """
        Retrieve annotation information for one sample frame.

        Returns:
            group_label: Group activity label for this sample.
            frame_boxes: Dictionary of annotated frames and player boxes.
        """
        frame_info = self.videos_annot[video_id][frame_id]

        group_label = frame_info["category"]
        frame_boxes = frame_info["frame_boxes_dct"]

        return group_label, frame_boxes

    def _iter_selected_frames(self, videos_id):
        """
        Iterate over annotations belonging to the selected videos.

        The selected IDs are converted to a set for efficient membership
        checks. The annotation dictionary determines the iteration order.

        Supports integer or string video IDs when their string
        representations match.
        """
        selected_video_ids = {str(video_id) for video_id in videos_id}

        for video_id, video_info in self.videos_annot.items():
            if str(video_id) not in selected_video_ids:
                continue

            for frame_id, frame_info in video_info.items():
                yield video_id, frame_id, frame_info

    def image_classification(self, videos_id):
        """
        Baseline 1: Full-image group activity classification.

        One sample contains:
            [image_path, group_label]

        The model receives the entire frame and predicts one of
        the group activity classes.
        """
        dataset = {}

        for index, (video_id, frame_id, frame_info) in enumerate(
            self._iter_selected_frames(videos_id)
        ):
            # Locate the image associated with this annotated frame.
            image_path = self._get_image_path(video_id, frame_id)

            # The target is the activity of the entire group.
            group_label = frame_info["category"]

            dataset[index] = [image_path, group_label]

        return dataset

    def cropped_player_classification(self, videos_id):
        """
        Baseline 2: Individual player action classification.

        One sample contains:
            [image_path, [x1, y1, x2, y2], player_label]

        Each annotated player becomes a separate training sample.
        Several samples can therefore refer to the same source image.
        """
        dataset = {}
        index = 0

        for video_id, frame_id, frame_info in self._iter_selected_frames(
            videos_id
        ):
            image_path = self._get_image_path(video_id, frame_id)

            # Retrieve the player annotations for this frame.
            frame_boxes = frame_info["frame_boxes_dct"]
            players = frame_boxes[int(frame_id)]

            for player in players:
                # Bounding box coordinates define the player's crop.
                x1, y1, x2, y2 = player.box

                # The target is this player's action, not the group action.
                player_label = player.category

                dataset[index] = [
                    image_path,
                    [x1, y1, x2, y2],
                    player_label,
                ]
                index += 1

        return dataset

    def cropped_player_team_with_group_classification(self, videos_id):
        """
        Baseline 3: Group activity classification using player crops.

        One sample contains:
            [box_1, box_2, ..., box_N, group_label, image_path]

        Each bounding box represents one player in the same frame.
        The target remains the group activity label.

        The number of players N can vary between samples.
        Padding and masking should be handled by collate.py.
        """
        dataset = {}
        index = 0

        for video_id, frame_id, frame_info in self._iter_selected_frames(
            videos_id
        ):
            image_path = self._get_image_path(video_id, frame_id)
            group_label = frame_info["category"]

            players = frame_info["frame_boxes_dct"][int(frame_id)]

            # Collect the bounding boxes of all players in this frame.
            boxes = []

            for player in players:
                x1, y1, x2, y2 = player.box
                boxes.append([x1, y1, x2, y2])

            # Keep the label and image path at the end because the
            # corresponding DatasetBaseline expects this exact structure.
            dataset[index] = [
                *boxes,
                group_label,
                image_path,
            ]
            index += 1

        return dataset

    def seq_of_image(self, videos_id):
        """
        Baseline 5: Group activity classification from image sequences.

        One sample contains:
            [frame_1_path, ..., frame_T_path, group_label]

        The frame paths come from frame_boxes_dct's keys.

        Important:
            The number and order of frames must be consistent with the
            intended temporal sequence. Dictionary insertion order alone
            does not guarantee chronological ordering.
        """
        dataset = {}
        index = 0

        for video_id, frame_id, frame_info in self._iter_selected_frames(
            videos_id
        ):
            group_label = frame_info["category"]
            frame_boxes = frame_info["frame_boxes_dct"]

            # Sort frame IDs numerically to maintain chronological order.
            sequence_frame_ids = sorted(
                frame_boxes.keys(),
                key=lambda value: int(value),
            )

            # Each path uses the sample's directory but the actual
            # sequence frame ID as the image filename.
            frame_paths = [
                self._get_sequence_image_path(
                    video_id=video_id,
                    sample_frame_id=frame_id,
                    sequence_frame_id=sequence_frame_id,
                )
                for sequence_frame_id in sequence_frame_ids
            ]

            # Append the group target after all frame paths.
            dataset[index] = [*frame_paths, group_label]
            index += 1

        return dataset

    def _get_sequence_image_path(
        self,
        video_id,
        sample_frame_id,
        sequence_frame_id,
    ):
        """
        Build a path for one frame inside a temporal sequence.

        This preserves the original directory convention:
            directory = sample_frame_id
            filename  = sequence_frame_id.jpg

        Verify that this matches the real dataset layout before training.
        """
        return os.path.join(
            self.videos_root,
            str(video_id),
            str(sample_frame_id),
            f"{sequence_frame_id}.jpg",
        )

    def seq_of_cropped_player_team_with_group_classification(self, videos_id):
        """
        Baseline 6: Group activity classification using player sequences.

        One sample contains:
            [
                frame_1_info,
                ...,
                frame_T_info,
                group_label,
            ]

        Each frame_info contains:
            [image_path, box_1, box_2, ..., box_N]

        Each player box is stored as:
            (x1, y1, x2, y2, player_label)

        Players are sorted by their leftmost x-coordinate within each
        frame. This provides a deterministic spatial ordering, but it
        does NOT guarantee that a player keeps the same identity across
        frames. Player tracking or identity matching is a separate task.

        The number of detections can differ between frames.
        """
        dataset = {}
        index = 0

        for video_id, frame_id, frame_info in self._iter_selected_frames(
            videos_id
        ):
            group_label = frame_info["category"]
            frame_boxes = frame_info["frame_boxes_dct"]

            # Sort frame IDs numerically to preserve temporal order.
            sequence_frame_ids = sorted(
                frame_boxes.keys(),
                key=lambda value: int(value),
            )

            sequence_data = []

            for sequence_frame_id in sequence_frame_ids:
                # Use the annotation frame's directory, and the current
                # sequence frame ID for the actual image filename.
                image_path = self._get_sequence_image_path(
                    video_id=video_id,
                    sample_frame_id=frame_id,
                    sequence_frame_id=sequence_frame_id,
                )

                players = frame_boxes[sequence_frame_id]

                # Sort detections from left to right for reproducibility.
                sorted_players = sorted(
                    players,
                    key=lambda player: player.box[0],
                )

                # The first element identifies the image for this frame.
                frame_data = [image_path]

                for player in sorted_players:
                    x1, y1, x2, y2 = player.box
                    player_label = player.category

                    # Keep the action label in the sample, even if a
                    # particular model currently uses only the box.
                    frame_data.append(
                        (x1, y1, x2, y2, player_label)
                    )

                sequence_data.append(frame_data)

            # Store the group label after the complete sequence.
            dataset[index] = [*sequence_data, group_label]
            index += 1

        return dataset
