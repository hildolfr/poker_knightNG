import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import jsonschema


ROOT = Path(__file__).parents[2]


def request_bytes(backend: str) -> bytes:
    raw = {
        "contract_version": "v1",
        "hero_cards": ["As", "Ah"],
        "board_cards": ["2s", "3h", "Td"],
        "opponent_count": "2",
        "requested_trials": "1",
        "seed": "0x0123456789abcdef",
        "backend": backend,
        "rng": {
            "algorithm_id": "poker-knight-ng/philox4x32-10",
            "algorithm_version": "1",
        },
    }
    return json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def venv_python(venv: Path) -> Path:
    return venv / "bin/python"


def test_main_module_is_import_safe():
    imported = subprocess.run(
        [sys.executable, "-c", "import poker_knight_ng.__main__; print('ok')"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert (imported.returncode, imported.stdout, imported.stderr) == (
        0,
        "ok\n",
        "",
    )


def test_wheel_and_sdist_include_and_install_the_cli(tmp_path):
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--out-dir", str(dist)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    wheel, = tuple(dist.glob("*.whl"))
    sdist, = tuple(dist.glob("*.tar.gz"))

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    assert "poker_knight_ng/cli.py" in wheel_names
    assert "poker_knight_ng/__main__.py" in wheel_names
    assert "poker_knight_ng/contract/serialize.py" in wheel_names
    entry_points, = tuple(
        name for name in wheel_names if name.endswith(".dist-info/entry_points.txt")
    )
    with zipfile.ZipFile(wheel) as archive:
        assert "poker-knight-ng = poker_knight_ng.cli:main" in (
            archive.read(entry_points).decode()
        )

    with tarfile.open(sdist) as archive:
        sdist_names = set(archive.getnames())
    assert any(name.endswith("/src/poker_knight_ng/cli.py") for name in sdist_names)
    assert any(name.endswith("/src/poker_knight_ng/__main__.py") for name in sdist_names)
    assert any(
        name.endswith("/src/poker_knight_ng/contract/serialize.py")
        for name in sdist_names
    )

    result_schema = json.loads(
        (ROOT / "contracts/v1/equity-result.schema.json").read_text()
    )
    problem_schema = json.loads(
        (ROOT / "contracts/v1/problem.schema.json").read_text()
    )
    for index, artifact in enumerate((wheel, sdist)):
        venv = tmp_path / f"venv-{index}"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(venv)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python(venv)),
                str(artifact),
            ],
            check=True,
            capture_output=True,
        )
        console = venv / "bin/poker-knight-ng"
        assert console.is_file()
        no_cupy = subprocess.run(
            [
                str(venv_python(venv)),
                "-c",
                "import importlib.util; assert importlib.util.find_spec('cupy') is None",
            ],
            capture_output=True,
        )
        assert no_cupy.returncode == 0

        cpu = subprocess.run(
            [str(console), "solve"],
            input=request_bytes("cpu_reference"),
            capture_output=True,
        )
        assert (cpu.returncode, cpu.stderr) == (0, b"")
        jsonschema.Draft202012Validator(result_schema).validate(
            json.loads(cpu.stdout)
        )

        cuda = subprocess.run(
            [str(console), "solve-cuda"],
            input=request_bytes("cuda"),
            capture_output=True,
        )
        assert (cuda.returncode, cuda.stdout) == (4, b"")
        problem = json.loads(cuda.stderr)
        jsonschema.Draft202012Validator(problem_schema).validate(problem)
        assert problem["code"] == "BACKEND_UNAVAILABLE"
