#!/usr/bin/env python3
"""Build the vetted 74-episode Quest3/Franka dual-camera dataset.

The output contains the previously vetted 50-episode dataset followed by the
24 vetted episodes recorded after the wrist-camera repositioning.  Parquet
episode/global indices and all LeRobot v2.1 metadata are rewritten so this is
one coherent dataset, rather than a directory-level concatenation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


NEW_SOURCE_IDS = (
    "0121",
    "0124",
    "0126",
    "0127",
    "0128",
    "0131",
    "0132",
    "0134",
    "0135",
    "0136",
    "0137",
    "0138",
    "0139",
    "0140",
    "0142",
    "0143",
    "0144",
    "0145",
    "0146",
    "0147",
    "0148",
    "0149",
    "0150",
    "0151",
)

EXCLUDED_SOURCE_IDS = {
    "0036": "excluded from the original vetted 50 episodes",
    "0125": "incomplete recording",
    "0129": "frozen wrist-camera video",
    "0130": "incomplete recording",
    "0133": "incomplete recording",
    "0141": "video/parquet frame-count mismatch",
    "0152": "incomplete recording",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, separators=(",", ":")) + "\n")


def feature_signature(info: dict) -> dict:
    return {
        key: {
            "dtype": value.get("dtype"),
            "shape": value.get("shape"),
            "names": value.get("names"),
        }
        for key, value in info["features"].items()
    }


def locate_episode_file(dataset: Path, template: str, episode_index: int) -> Path:
    info = load_json(dataset / "meta" / "info.json")
    relative = template.format(
        episode_chunk=episode_index // info["chunks_size"],
        episode_index=episode_index,
    )
    path = dataset / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def set_integer_column(table: pa.Table, name: str, value: int, *, sequence: bool = False) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Required parquet column is missing: {name}")
    column_type = table.schema.field(column_index).type
    if sequence:
        values = pa.array(range(value, value + table.num_rows), type=column_type)
    else:
        values = pa.array([value] * table.num_rows, type=column_type)
    return table.set_column(column_index, name, values)


def schemas_are_compatible(candidate: pa.Schema, canonical: pa.Schema) -> bool:
    """Accept legacy variable lists when the canonical field is a fixed list."""
    if candidate.names != canonical.names:
        return False
    for candidate_field, canonical_field in zip(candidate, canonical, strict=True):
        candidate_type = candidate_field.type
        canonical_type = canonical_field.type
        if candidate_type == canonical_type:
            continue
        if (
            (pa.types.is_list(candidate_type) or pa.types.is_fixed_size_list(candidate_type))
            and pa.types.is_fixed_size_list(canonical_type)
            and candidate_type.value_type == canonical_type.value_type
        ):
            continue
        return False
    return True


def normalize_table_schema(table: pa.Table, canonical: pa.Schema) -> pa.Table:
    """Convert legacy list columns to the fixed-size LeRobot v2.1 schema."""
    arrays = []
    for field in canonical:
        column = table[field.name].combine_chunks()
        if column.type == field.type:
            arrays.append(column)
        else:
            arrays.append(pa.array(column.to_pylist(), type=field.type))
    return pa.Table.from_arrays(arrays, schema=canonical)


def build_dataset(data_root: Path, output: Path) -> None:
    base = data_root / "quest3_franka_dualcam_pickplace_50eps"
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if staging.exists():
        raise FileExistsError(f"Remove or inspect stale staging directory first: {staging}")

    base_info = load_json(base / "meta" / "info.json")
    base_episodes = load_jsonl(base / "meta" / "episodes.jsonl")
    base_stats = load_jsonl(base / "meta" / "episodes_stats.jsonl")
    if base_info["total_episodes"] != 50 or len(base_episodes) != 50 or len(base_stats) != 50:
        raise ValueError("The vetted base dataset is not the expected complete 50-episode dataset")

    base_manifest = load_json(base / "meta" / "merge_manifest.json")
    old_source_ids = base_manifest.get("source_ids", [])
    if len(old_source_ids) != 50 or "0036" in old_source_ids:
        raise ValueError("The base merge manifest is not the expected vetted source list")

    source_entries: list[tuple[Path, int, str]] = []
    for old_index, source_id in enumerate(old_source_ids):
        source_entries.append((base, old_index, source_id))
    for source_id in NEW_SOURCE_IDS:
        source_entries.append(
            (data_root / f"quest3_franka_dualcam_test_{source_id}", 0, source_id)
        )
    if len(source_entries) != 74:
        raise AssertionError(f"Expected 74 sources, got {len(source_entries)}")

    reference_features = feature_signature(base_info)
    canonical_dataset = data_root / f"quest3_franka_dualcam_test_{NEW_SOURCE_IDS[0]}"
    canonical_info = load_json(canonical_dataset / "meta" / "info.json")
    canonical_parquet = locate_episode_file(canonical_dataset, canonical_info["data_path"], 0)
    canonical_schema = pq.ParquetFile(canonical_parquet).schema_arrow
    task_text = "pick up the cube and place it on the box"
    validated: list[tuple[Path, int, str, dict, dict, Path, dict[str, Path]]] = []

    for dataset, old_index, source_id in source_entries:
        info_path = dataset / "meta" / "info.json"
        episodes_path = dataset / "meta" / "episodes.jsonl"
        stats_path = dataset / "meta" / "episodes_stats.jsonl"
        for required in (info_path, episodes_path, stats_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        info = load_json(info_path)
        if info.get("codebase_version") != "v2.1" or info.get("fps") != base_info["fps"]:
            raise ValueError(f"Incompatible LeRobot version/FPS in {dataset}")
        if feature_signature(info) != reference_features:
            raise ValueError(f"Feature signature differs from the base dataset: {dataset}")

        episode_records = {item["episode_index"]: item for item in load_jsonl(episodes_path)}
        stats_records = {item["episode_index"]: item for item in load_jsonl(stats_path)}
        episode = episode_records.get(old_index)
        stats = stats_records.get(old_index)
        if episode is None or stats is None:
            raise ValueError(f"Missing episode/stats record {old_index} in {dataset}")
        if episode.get("tasks") != [task_text]:
            raise ValueError(f"Unexpected task annotation in {dataset}: {episode.get('tasks')}")

        parquet_path = locate_episode_file(dataset, info["data_path"], old_index)
        parquet_file = pq.ParquetFile(parquet_path)
        if parquet_file.metadata.num_rows != episode["length"]:
            raise ValueError(
                f"Parquet length mismatch in {dataset}: "
                f"{parquet_file.metadata.num_rows} != {episode['length']}"
            )
        if not schemas_are_compatible(parquet_file.schema_arrow, canonical_schema):
            raise ValueError(f"Parquet schema is incompatible with the canonical dataset: {dataset}")

        videos: dict[str, Path] = {}
        for key, feature in info["features"].items():
            if feature.get("dtype") != "video":
                continue
            relative = info["video_path"].format(
                episode_chunk=old_index // info["chunks_size"],
                episode_index=old_index,
                video_key=key,
            )
            video_path = dataset / relative
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise FileNotFoundError(video_path)
            videos[key] = video_path
        if len(videos) != 2:
            raise ValueError(f"Expected two videos in {dataset}, got {sorted(videos)}")

        validated.append((dataset, old_index, source_id, episode, stats, parquet_path, videos))

    (staging / "meta").mkdir(parents=True)
    (staging / "data" / "chunk-000").mkdir(parents=True)
    all_episodes: list[dict] = []
    all_stats: list[dict] = []
    manifest_episodes: list[dict] = []
    global_index = 0

    for new_index, (dataset, old_index, source_id, episode, stats, parquet_path, videos) in enumerate(validated):
        new_episode = dict(episode)
        new_episode["episode_index"] = new_index
        all_episodes.append(new_episode)

        new_stats = dict(stats)
        new_stats["episode_index"] = new_index
        all_stats.append(new_stats)

        table = normalize_table_schema(pq.read_table(parquet_path), canonical_schema)
        table = set_integer_column(table, "episode_index", new_index)
        table = set_integer_column(table, "index", global_index, sequence=True)
        table = set_integer_column(table, "task_index", 0)
        destination = staging / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet"
        pq.write_table(table, destination)

        for video_key, video_path in videos.items():
            video_destination = (
                staging
                / "videos"
                / "chunk-000"
                / video_key
                / f"episode_{new_index:06d}.mp4"
            )
            video_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, video_destination)

        manifest_episodes.append(
            {
                "episode_index": new_index,
                "source_id": source_id,
                "source_dataset": dataset.name,
                "source_episode_index": old_index,
                "length": table.num_rows,
            }
        )
        global_index += table.num_rows

    output_info = dict(base_info)
    output_info.update(
        {
            "total_episodes": len(all_episodes),
            "total_frames": global_index,
            "total_tasks": 1,
            "total_videos": 2 * len(all_episodes),
            "total_chunks": 1,
            "splits": {"train": f"0:{len(all_episodes)}"},
        }
    )
    write_json(staging / "meta" / "info.json", output_info)
    write_jsonl(staging / "meta" / "episodes.jsonl", all_episodes)
    write_jsonl(staging / "meta" / "episodes_stats.jsonl", all_stats)
    write_jsonl(
        staging / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": task_text}],
    )
    shutil.copy2(base / "meta" / "modality.json", staging / "meta" / "modality.json")
    write_json(
        staging / "meta" / "merge_manifest.json",
        {
            "output_repo_id": f"snkdjn/{output.name}",
            "task": task_text,
            "base_dataset": base.name,
            "base_episode_count": 50,
            "new_wrist_camera_episode_count": len(NEW_SOURCE_IDS),
            "source_ids": list(old_source_ids) + list(NEW_SOURCE_IDS),
            "new_wrist_camera_source_ids": list(NEW_SOURCE_IDS),
            "excluded_source_ids": EXCLUDED_SOURCE_IDS,
            "episodes": manifest_episodes,
        },
    )

    if len(list((staging / "data" / "chunk-000").glob("*.parquet"))) != 74:
        raise RuntimeError("Final parquet-file count is not 74")
    if len(list((staging / "videos" / "chunk-000").glob("*/*.mp4"))) != 148:
        raise RuntimeError("Final video-file count is not 148")
    staging.rename(output)
    print(f"Built {output}")
    print(f"Episodes: {len(all_episodes)}")
    print(f"Frames: {global_index}")
    print(f"Videos: {2 * len(all_episodes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/dase-hw101/franka_ws/dataset/snkdjn"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/dase-hw101/franka_ws/dataset/snkdjn/"
            "quest3_franka_dualcam_pickplace_74eps"
        ),
    )
    args = parser.parse_args()
    build_dataset(args.data_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
