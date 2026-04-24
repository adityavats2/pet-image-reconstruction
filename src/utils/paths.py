from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_dir: Path
    data_dir: Path
    results_dir: Path
    checkpoints_dir: Path


def resolve_project_dir(start_path: Path | None = None) -> Path:
    """Locate the repo root from a file path or the current working directory."""
    candidates: list[Path] = []

    if start_path is not None:
        resolved = start_path.resolve()
        candidates.append(resolved.parent if resolved.is_file() else resolved)

    candidates.append(Path.cwd().resolve())

    for start in candidates:
        for path in [start, *start.parents]:
            if (path / "requirements.txt").exists() and (path / "notebooks").exists():
                return path

    return candidates[0]


def get_project_paths(start_path: Path | None = None) -> ProjectPaths:
    project_dir = resolve_project_dir(start_path)
    paths = ProjectPaths(
        project_dir=project_dir,
        data_dir=project_dir / "data",
        results_dir=project_dir / "results",
        checkpoints_dir=project_dir / "checkpoints",
    )

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return paths
