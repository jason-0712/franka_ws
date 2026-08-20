#!/usr/bin/env python3
"""Build the fixed 74+20 Quest3/Franka replay dataset.

The first 74 episodes are copied from the previously vetted merged dataset.
Twenty unique spatial episodes are then appended from their original source
datasets.  Episode/global indices and LeRobot v2.1 metadata are rewritten so
the output is one coherent 94-episode dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


EXPECTED_APPEND_GROUPS = ("front", "back")
EXPECTED_PER_APPEND_GROUP = 10
EXPECTED_BASE_EPISODES = 74
EXPECTED_TOTAL_EPISODES = 94


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def set_integer_column(
    table: pa.Table, name: str, value: int, *, sequence: bool = False
) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Required parquet column is missing: {name}")
    column_type = table.schema.field(column_index).type
    values = (
        pa.array(range(value, value + table.num_rows), type=column_type)
        if sequence
        else pa.array([value] * table.num_rows, type=column_type)
    )
    return table.set_column(column_index, name, values)


def schemas_are_compatible(candidate: pa.Schema, canonical: pa.Schema) -> bool:
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
    arrays = []
    for field in canonical:
        column = table[field.name].combine_chunks()
        arrays.append(
            column if column.type == field.type else pa.array(column.to_pylist(), type=field.type)
        )
    return pa.Table.from_arrays(arrays, schema=canonical)


def flatten_append_entries(manifest: dict) -> list[dict]:
    groups = manifest.get("append_groups", [])
    names = tuple(group.get("name") for group in groups)
    if names != EXPECTED_APPEND_GROUPS:
        raise ValueError(f"Expected append groups {EXPECTED_APPEND_GROUPS}, got {names}")

    entries: list[dict] = []
    for group in groups:
        episodes = group.get("episodes", [])
        if len(episodes) != EXPECTED_PER_APPEND_GROUP:
            raise ValueError(
                f"Group {group['name']} must contain exactly "
                f"{EXPECTED_PER_APPEND_GROUP} episodes"
            )
        for episode in episodes:
            entry = dict(episode)
            entry["position_group"] = group["name"]
            entries.append(entry)

    identities = [
        (entry["source_dataset"], int(entry["source_episode_index"]))
        for entry in entries
    ]
    if len(entries) != 20 or len(set(identities)) != 20:
        raise ValueError("Manifest must describe exactly 20 unique append episodes")
    return entries


def build_dataset(data_root: Path, output: Path, manifest_path: Path) -> None:
    manifest = load_json(manifest_path)
    append_entries = flatten_append_entries(manifest)
    expected_output_name = manifest["output_repo_id"].split("/", 1)[-1]
    if output.name != expected_output_name:
        raise ValueError(
            f"Output directory name must match manifest: "
            f"{output.name} != {expected_output_name}"
        )

    base = data_root / manifest["base_dataset"]
    base_info = load_json(base / "meta" / "info.json")
    base_episodes = load_jsonl(base / "meta" / "episodes.jsonl")
    base_stats = load_jsonl(base / "meta" / "episodes_stats.jsonl")
    base_lineage = load_json(base / "meta" / "merge_manifest.json")
    if (
        base_info.get("total_episodes") != EXPECTED_BASE_EPISODES
        or len(base_episodes) != EXPECTED_BASE_EPISODES
        or len(base_stats) != EXPECTED_BASE_EPISODES
        or len(base_lineage.get("episodes", [])) != EXPECTED_BASE_EPISODES
    ):
        raise ValueError("Base dataset is not the expected vetted 74-episode dataset")

    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if staging.exists():
        raise FileExistsError(f"Remove or inspect stale staging directory first: {staging}")

    canonical_parquet = locate_episode_file(base, base_info["data_path"], 0)
    canonical_schema = pq.ParquetFile(canonical_parquet).schema_arrow
    reference_features = feature_signature(base_info)
    task_text = manifest["task"]

    entries: list[dict] = []
    for base_index, lineage in enumerate(base_lineage["episodes"]):
        if int(lineage["episode_index"]) != base_index:
            raise ValueError("Base lineage episode indices are not contiguous")
        entries.append(
            {
                "source_kind": "base74",
                "position_group": "base74",
                "dataset": base,
                "source_episode_index": base_index,
                "source_id": lineage["source_id"],
                "lineage_source_dataset": lineage["source_dataset"],
                "lineage_source_episode_index": lineage["source_episode_index"],
            }
        )
    for entry in append_entries:
        entries.append(
            {
                "source_kind": "spatial_append",
                "position_group": entry["position_group"],
                "dataset": data_root / entry["source_dataset"],
                "source_episode_index": int(entry["source_episode_index"]),
                "source_id": entry["source_id"],
                "lineage_source_dataset": entry["source_dataset"],
                "lineage_source_episode_index": int(entry["source_episode_index"]),
            }
        )
    if len(entries) != EXPECTED_TOTAL_EPISODES:
        raise AssertionError(f"Expected 94 sources, got {len(entries)}")

    append_identities = {
        (entry["lineage_source_dataset"], entry["lineage_source_episode_index"])
        for entry in entries
        if entry["source_kind"] == "spatial_append"
    }
    base_identities = {
        (entry["lineage_source_dataset"], entry["lineage_source_episode_index"])
        for entry in entries
        if entry["source_kind"] == "base74"
    }
    if append_identities & base_identities:
        raise ValueError("Append set duplicates one or more base-74 source identities")

    validated: list[dict] = []
    for entry in entries:
        dataset = entry["dataset"]
        old_index = entry["source_episode_index"]
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
            raise ValueError(f"Feature signature differs from base dataset: {dataset}")

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
            raise ValueError(f"Incompatible parquet schema in {dataset}")

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

        validated.append(
            {
                **entry,
                "episode": episode,
                "stats": stats,
                "parquet_path": parquet_path,
                "videos": videos,
            }
        )

    (staging / "meta").mkdir(parents=True)
    (staging / "data" / "chunk-000").mkdir(parents=True)
    all_episodes: list[dict] = []
    all_stats: list[dict] = []
    output_manifest_episodes: list[dict] = []
    global_index = 0

    for new_index, item in enumerate(validated):
        new_episode = dict(item["episode"])
        new_episode["episode_index"] = new_index
        all_episodes.append(new_episode)

        new_stats = dict(item["stats"])
        new_stats["episode_index"] = new_index
        all_stats.append(new_stats)

        table = normalize_table_schema(pq.read_table(item["parquet_path"]), canonical_schema)
        table = set_integer_column(table, "episode_index", new_index)
        table = set_integer_column(table, "index", global_index, sequence=True)
        table = set_integer_column(table, "task_index", 0)
        parquet_destination = (
            staging / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet"
        )
        pq.write_table(table, parquet_destination)

        video_hashes = {}
        for video_key, video_path in item["videos"].items():
            video_destination = (
                staging
                / "videos"
                / "chunk-000"
                / video_key
                / f"episode_{new_index:06d}.mp4"
            )
            video_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, video_destination)
            video_hashes[video_key] = sha256(video_path)

        output_manifest_episodes.append(
            {
                "episode_index": new_index,
                "source_kind": item["source_kind"],
                "position_group": item["position_group"],
                "source_id": item["source_id"],
                "source_dataset": item["lineage_source_dataset"],
                "source_episode_index": item["lineage_source_episode_index"],
                "immediate_dataset": item["dataset"].name,
                "immediate_episode_index": item["source_episode_index"],
                "length": table.num_rows,
                "source_parquet_sha256": sha256(item["parquet_path"]),
                "source_video_sha256": video_hashes,
            }
        )
        global_index += table.num_rows

    output_info = dict(base_info)
    output_info.update(
        {
            "total_episodes": EXPECTED_TOTAL_EPISODES,
            "total_frames": global_index,
            "total_tasks": 1,
            "total_videos": EXPECTED_TOTAL_EPISODES * 2,
            "total_chunks": 1,
            "splits": {"train": f"0:{EXPECTED_TOTAL_EPISODES}"},
        }
    )
    write_json(staging / "meta" / "info.json", output_info)
    write_jsonl(staging / "meta" / "episodes.jsonl", all_episodes)
    write_jsonl(staging / "meta" / "episodes_stats.jsonl", all_stats)
    write_jsonl(staging / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task_text}])
    shutil.copy2(base / "meta" / "modality.json", staging / "meta" / "modality.json")
    shutil.copy2(manifest_path, staging / "meta" / "source_selection_manifest.json")
    write_json(
        staging / "meta" / "merge_manifest.json",
        {
            "schema_version": 1,
            "output_repo_id": manifest["output_repo_id"],
            "task": task_text,
            "design": manifest["design"],
            "base_dataset": base.name,
            "base_episode_count": EXPECTED_BASE_EPISODES,
            "append_group_counts": {"front": 10, "back": 10},
            "total_episode_count": EXPECTED_TOTAL_EPISODES,
            "source_selection_manifest": manifest_path.name,
            "source_selection_manifest_sha256": sha256(manifest_path),
            "holdout_policy": manifest.get("holdout_policy", {}),
            "episodes": output_manifest_episodes,
        },
    )

    parquet_count = len(list((staging / "data" / "chunk-000").glob("*.parquet")))
    video_count = len(list((staging / "videos" / "chunk-000").glob("*/*.mp4")))
    if parquet_count != EXPECTED_TOTAL_EPISODES or video_count != 188:
        raise RuntimeError(
            f"Final file count mismatch: parquet={parquet_count}, video={video_count}"
        )
    staging.rename(output)
    print(f"Built: {output}")
    print("Groups: base74=74, front=10, back=10")
    print("Episodes: 94")
    print(f"Frames: {global_index}")
    print("Videos: 188")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    build_dataset(args.data_root.resolve(), args.output.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()
