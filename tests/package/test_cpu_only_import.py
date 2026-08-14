import importlib.resources
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


LEGACY_PATHS = (
    "api.py",
    "api_server.py",
    "memory_manager.py",
    "gpu_structures.py",
    "result_builder.py",
    "card_utils.py",
    "validator.py",
    "cuda",
)
REQUIRED_PYTHON = (
    "__init__.py",
    "contract/__init__.py",
    "contract/canonical.py",
    "contract/errors.py",
    "contract/models.py",
    "schemas/__init__.py",
    "schemas/v1/__init__.py",
)


def test_root_contract_import_never_loads_cupy_or_cuda_compiler():
    code = "import poker_knight_ng; import sys; assert 'cupy' not in sys.modules; assert not any('cuda.kernel_wrapper' in x for x in sys.modules); print(poker_knight_ng.__version__)"
    completed = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert completed.stdout.strip()
    assert importlib.util.find_spec("cupy") is None
    assert importlib.util.find_spec("poker_knight_ng.cuda") is None


def test_inert_cuda_sources_are_resources_not_a_python_namespace():
    root = importlib.resources.files("poker_knight_ng")
    assert root.joinpath("cuda-sources/poker_kernel.cu").read_text()
    assert importlib.util.find_spec("poker_knight_ng.cuda") is None


def test_installed_schema_resources_are_authoritative_copies():
    root = Path(__file__).parents[2]
    for name in ("equity-request.schema.json", "equity-result.schema.json", "problem.schema.json"):
        installed = importlib.resources.files("poker_knight_ng.schemas.v1").joinpath(name).read_bytes()
        assert installed == (root / "contracts" / "v1" / name).read_bytes()


def _archive_members(path: Path) -> set[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return set(archive.namelist())
    with tarfile.open(path) as archive:
        return set(archive.getnames())


def test_contract_only_source_and_archives_exclude_legacy_and_test_content(tmp_path):
    root = Path(__file__).parents[2]
    package = root / "src" / "poker_knight_ng"

    assert not [path for path in LEGACY_PATHS if (package / path).exists()]
    for module in (
        "poker_knight_ng.api",
        "poker_knight_ng.api_server",
        "poker_knight_ng.memory_manager",
        "poker_knight_ng.gpu_structures",
        "poker_knight_ng.result_builder",
        "poker_knight_ng.card_utils",
        "poker_knight_ng.validator",
        "poker_knight_ng.cuda",
    ):
        assert importlib.util.find_spec(module) is None

    required_native = {path.relative_to(package).as_posix() for suffix in ("*.cu", "*.cuh") for path in package.rglob(suffix)}
    standalone_evaluator = {"cuda-sources/cards.cuh", "cuda-sources/evaluator.cuh"}
    standalone_dealer = {"cuda-sources/philox.cuh", "cuda-sources/dealer.cuh"}
    standalone_simulator = {"cuda-sources/simulate.cuh"}
    standalone_reducer = {"cuda-sources/reduce.cuh"}
    standalone_kernels = {"cuda-sources/deterministic_kernels.cu"}
    assert standalone_evaluator <= required_native
    assert standalone_dealer <= required_native
    assert standalone_simulator <= required_native
    assert standalone_reducer <= required_native
    assert standalone_kernels <= required_native
    assert len(required_native) == 13
    assert all((package / path).is_file() for path in REQUIRED_PYTHON)

    subprocess.run(["uv", "build", "--out-dir", str(tmp_path)], cwd=root, check=True)
    artifacts = sorted(path for path in tmp_path.iterdir() if path.name.endswith((".tar.gz", ".whl")))
    assert len(artifacts) == 2
    for artifact in artifacts:
        members = _archive_members(artifact)
        normalized = {member.split("/", 1)[-1] if artifact.suffix == ".gz" else member for member in members}
        package_prefix = "poker_knight_ng/" if artifact.suffix == ".whl" else "src/poker_knight_ng/"
        assert not any(member.endswith(path) for member in normalized for path in LEGACY_PATHS)
        assert not any("__pycache__/" in member or member.endswith(".pyc") for member in normalized)
        assert not any(member.startswith(("tests/", "benchmarks/")) or "/tests/" in member or "/benchmarks/" in member for member in normalized)
        assert all(f"{package_prefix}{path}" in normalized for path in REQUIRED_PYTHON)
        assert all(f"{package_prefix}{path}" in normalized for path in required_native)
