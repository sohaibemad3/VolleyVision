
"""
Load and organize annotations for the Volleyball Group Activity
Recognition dataset.

This module:
    1. Loads player tracking annotations from text files.
    2. Loads group activity labels for each clip.
    3. Builds a nested dictionary containing videos, clips, and frames.
    4. Saves and loads the processed annotations as a pickle file.

Expected output structure:
    videos_annot[video_id][clip_id] = {
        "category": group_activity_label,
        "frame_boxes_dct": {
            frame_id: [BoxInfo, ...],
            ...
        },
    }

Note:
    BoxInfo must be defined or imported from the module that implements
    parsing of individual tracking annotation lines.
"""

import pickle
from pathlib import Path
import BoxInfo


def load_tracking_annot(
    annotation_path,
    max_players=12,
    trim_start=5,
    trim_end=6,
):
    """
    Load player tracking annotations for one clip.

    Each line in the tracking file is parsed into a BoxInfo object.
    Bounding boxes are first grouped by player ID so that the trimming
    operation is applied independently to each player's trajectory.

    Args:
        annotation_path: Path to the tracking annotation text file.
        max_players: Ignore player IDs outside [0, max_players - 1].
        trim_start: Number of initial boxes to discard per player.
        trim_end: Number of final boxes to discard per player.

    Returns:
        frame_boxes_dct:
            Dictionary mapping each frame ID to a list of BoxInfo
            objects belonging to that frame.

    Important:
        Trimming is performed separately for each player, not globally
        across all frames. The original annotation order is preserved
        within each player's list.
    """
    annotation_path = Path(annotation_path)

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Tracking annotation file not found: {annotation_path}"
        )

    if max_players <= 0:
        raise ValueError("max_players must be positive.")

    if trim_start < 0 or trim_end < 0:
        raise ValueError("trim_start and trim_end cannot be negative.")

    # Group each player's bounding boxes into a separate list.
    player_boxes = {
        player_id: []
        for player_id in range(max_players)
    }

    # Read and parse every line in the tracking annotation file.
    with annotation_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            # Ignore blank lines instead of trying to parse them.
            if not line.strip():
                continue

            # BoxInfo is responsible for parsing the annotation format.
            box_info = BoxInfo(line)

            # Ignore players whose IDs are outside the supported range.
            player_id = box_info.player_ID

            if not 0 <= player_id < max_players:
                continue

            player_boxes[player_id].append(box_info)

    # Reorganize the remaining boxes by frame ID.
    frame_boxes_dct = {}

    for boxes_info in player_boxes.values():
        # Remove the first trim_start annotations from this player.
        if trim_start:
            boxes_info = boxes_info[trim_start:]

        # Remove the final trim_end annotations without using [:-0],
        # which would incorrectly produce an empty list when trim_end=0.
        if trim_end:
            boxes_info = boxes_info[:-trim_end]

        # Add each retained box to its corresponding frame.
        for box_info in boxes_info:
            frame_id = box_info.frame_ID

            if frame_id not in frame_boxes_dct:
                frame_boxes_dct[frame_id] = []

            frame_boxes_dct[frame_id].append(box_info)

    return frame_boxes_dct


def load_video_annot(annotation_path):
    """
    Load group activity labels for all clips in one video.

    Expected annotation line:
        clip_filename.jpg group_activity_label

    Returns:
        Dictionary mapping clip IDs to group activity labels.

    Example:
        {
            "32564": "l-pass",
            "58624": "r-pass",
        }
    """
    annotation_path = Path(annotation_path)

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Video annotation file not found: {annotation_path}"
        )

    clip_category_dct = {}

    with annotation_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            # Skip empty lines.
            line = line.strip()

            if not line:
                continue

            # Split only on whitespace and require two fields:
            # the clip filename and its group activity label.
            items = line.split()

            if len(items) < 2:
                raise ValueError(
                    f"Invalid annotation at {annotation_path}:"
                    f"{line_number}: {line!r}"
                )

            clip_filename, category = items[:2]

            # Store clip IDs without the .jpg extension.
            clip_id = Path(clip_filename).stem
            clip_category_dct[clip_id] = category

    return clip_category_dct


def load_volleyball_dataset(videos_root, annot_root):
    """
    Load annotations for all videos in the Volleyball dataset.

    Directory layout expected:
        videos_root/
            video_id/
                annotations.txt
                clip_id/
                    frame images ...

        annot_root/
            video_id/
                clip_id/
                    clip_id.txt

    Returns:
        A nested dictionary indexed by video ID and clip ID.
        Each clip contains its group label and frame-level player boxes.

    This function loads annotation metadata; it does not load image pixels.
    """
    videos_root = Path(videos_root)
    annot_root = Path(annot_root)

    if not videos_root.is_dir():
        raise NotADirectoryError(
            f"Videos root directory not found: {videos_root}"
        )

    if not annot_root.is_dir():
        raise NotADirectoryError(
            f"Annotation root directory not found: {annot_root}"
        )

    videos_annot = {}

    # Sort video directories for reproducible ordering.
    video_dirs = sorted(
        path for path in videos_root.iterdir()
        if path.is_dir()
    )

    for video_dir in video_dirs:
        video_id = video_dir.name

        # Read the group activity label associated with each clip.
        video_annotation_path = video_dir / "annotations.txt"
        clip_category_dct = load_video_annot(video_annotation_path)

        clip_annot = {}

        # Each clip directory represents an annotated sample.
        clip_dirs = sorted(
            path for path in video_dir.iterdir()
            if path.is_dir()
        )

        for clip_dir in clip_dirs:
            clip_id = clip_dir.name

            # Ignore directories that do not have a group activity label.
            if clip_id not in clip_category_dct:
                raise ValueError(
                    f"Clip {clip_id!r} in video {video_id!r} "
                    "has no group activity label."
                )

            # Find the corresponding player tracking annotation file.
            tracking_path = (
                annot_root
                / video_id
                / clip_id
                / f"{clip_id}.txt"
            )

            # Convert tracking lines into frame-level player annotations.
            frame_boxes_dct = load_tracking_annot(tracking_path)

            # Store both the group label and its player annotations.
            clip_annot[clip_id] = {
                "category": clip_category_dct[clip_id],
                "frame_boxes_dct": frame_boxes_dct,
            }

        videos_annot[video_id] = clip_annot

    return videos_annot


def create_pkl_version(
    videos_root,
    annot_root,
    output_path,
):
    """
    Build the complete annotation dictionary and save it as a pickle file.

    Args:
        videos_root: Root directory containing video and clip directories.
        annot_root: Root directory containing tracking annotation files.
        output_path: Destination path for the .pkl file.

    Returns:
        The saved annotation dictionary.
    """
    output_path = Path(output_path)

    # Load and organize the complete annotation dataset.
    videos_annot = load_volleyball_dataset(
        videos_root=videos_root,
        annot_root=annot_root,
    )

    # Create the output directory if it does not already exist.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Pickle stores Python objects so they can be loaded without
    # reparsing all the original annotation text files.
    with output_path.open("wb") as file:
        pickle.dump(
            videos_annot,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"Annotations saved to: {output_path}")

    return videos_annot


def load_pkl_version(pkl_path):
    """
    Load a previously saved annotation dictionary from a pickle file.

    Only load pickle files from sources you trust because unpickling
    can execute arbitrary code.
    """
    pkl_path = Path(pkl_path)

    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"Pickle annotation file not found: {pkl_path}"
        )

    with pkl_path.open("rb") as file:
        videos_annot = pickle.load(file)

    return videos_annot


if __name__ == "__main__":
    # Replace these paths with your actual dataset locations.
    VIDEOS_ROOT = "/path/to/videos"
    ANNOT_ROOT = "/path/to/tracking_annotations"
    OUTPUT_PATH = "data/processed/volleyball_annotations.pkl"

    create_pkl_version(
        videos_root=VIDEOS_ROOT,
        annot_root=ANNOT_ROOT,
        output_path=OUTPUT_PATH,
    )