"""Data loading helpers for the local Charades-STA test split."""

from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True)
class STASample:
    video_id: str
    start: float
    end: float
    query: str
    video_path: Path
    duration: float


def load_video_lengths(csv_path):
    """Read Charades metadata and return video duration by id."""
    lengths = {}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lengths[row["id"]] = float(row["length"])
    return lengths


def load_sta_annotations(sta_path, video_dir, lengths, limit=None):
    """Parse Charades-STA lines: video_id start end ## sentence."""
    samples = []
    video_dir = Path(video_dir)
    with Path(sta_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            head, query = line.split("##", 1)
            video_id, start, end = head.split()
            sample = STASample(
                video_id=video_id,
                start=float(start),
                end=float(end),
                query=query.strip(),
                video_path=video_dir / f"{video_id}.mp4",
                duration=lengths[video_id],
            )
            samples.append(sample)

            if limit is not None and len(samples) >= limit:
                break

    return samples


def project_paths(project_root):
    """Centralize the local dataset layout used by notebooks and scripts."""
    root = Path(project_root)
    return {
        "sta_test": root / "Charades" / "Charades" / "charades_sta_test.txt",
        "test_csv": root / "Charades" / "Charades" / "Charades_v1_test.csv",
        "video_dir": root / "Charades" / "Charades_v1_480" / "Charades_v1_480",
    }
