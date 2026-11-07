"""The CI CPU fallback changes no backend unless an old x86 host is proven."""

from types import SimpleNamespace

import pytest

from tests import conftest as policy


@pytest.fixture
def host(monkeypatch):
    backend = SimpleNamespace(enabled=True, is_available=lambda: True)
    environment = {"CI": "true"}
    system = SimpleNamespace(platform="linux")
    platform = SimpleNamespace(machine=lambda: "x86_64")
    cpuinfo = SimpleNamespace(
        read_text=lambda *args, **kwargs: "processor: 0\nflags: fpu sse sse2 pni\n"
    )

    def cpuinfo_path(path):
        assert path == "/proc/cpuinfo"
        return cpuinfo

    def capability():
        raise AssertionError("DEFAULT capability is not a CPU flags probe")

    monkeypatch.setattr(
        policy,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(
                mkldnn=backend, cpu=SimpleNamespace(get_cpu_capability=capability)
            )
        ),
    )
    monkeypatch.setattr(policy, "os", SimpleNamespace(environ=environment))
    monkeypatch.setattr(policy, "sys", system)
    monkeypatch.setattr(policy, "platform", platform)
    monkeypatch.setattr(policy, "Path", cpuinfo_path)
    cleanups = []
    config = SimpleNamespace(add_cleanup=cleanups.append)
    return SimpleNamespace(
        backend=backend,
        config=config,
        cleanups=cleanups,
        environment=environment,
        system=system,
        platform=platform,
        cpuinfo=cpuinfo,
    )


@pytest.mark.parametrize("ci", ["true", "1"])
@pytest.mark.parametrize("machine", ["x86_64", "AMD64"])
def test_old_x86_ci_backend_is_disabled_and_restored(host, ci, machine):
    host.environment["CI"] = ci
    host.platform.machine = lambda: machine
    policy.pytest_configure(host.config)
    assert host.backend.enabled is False
    assert len(host.cleanups) == 1
    diagnostic = policy.pytest_report_header(host.config)
    assert "native Torch kernels" in diagnostic
    assert "MKLDNN disabled" in diagnostic
    assert "SSE4.1" in diagnostic
    host.cleanups[0]()
    assert host.backend.enabled is True


@pytest.mark.parametrize(
    "ci,system,machine",
    [
        ("", "linux", "x86_64"),
        ("false", "linux", "x86_64"),
        ("0", "linux", "x86_64"),
        ("true", "darwin", "x86_64"),
        ("1", "darwin", "arm64"),
        ("true", "linux", "aarch64"),
        ("true", "linux", "arm64"),
        ("1", "win32", "AMD64"),
    ],
)
def test_other_hosts_keep_their_backend(host, ci, system, machine):
    host.environment["CI"] = ci
    host.system.platform = system
    host.platform.machine = lambda: machine

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Other hosts must not inspect Linux CPU flags")

    host.cpuinfo.read_text = unexpected_read
    policy.pytest_configure(host.config)
    assert host.backend.enabled is True
    assert host.cleanups == []
    assert policy.pytest_report_header(host.config) is None


@pytest.mark.parametrize("available,enabled", [(False, True), (True, False)])
def test_unavailable_or_disabled_backend_is_preserved(host, available, enabled):
    host.backend.is_available = lambda: available
    host.backend.enabled = enabled

    def unexpected_read(*args, **kwargs):
        raise AssertionError("An inactive backend needs no CPU probe")

    host.cpuinfo.read_text = unexpected_read
    policy.pytest_configure(host.config)
    assert host.backend.enabled is enabled
    assert host.cleanups == []
    assert policy.pytest_report_header(host.config) is None


@pytest.mark.parametrize(
    "cpuinfo",
    [
        "",
        "processor: 0\nmodel name: unknown\n",
        "flags:\n",
        "flags: fpu unknown\n",
        "flags: fpu sse sse2 sse4_1\n",  # A non-AVX CPU can report DEFAULT.
        "flags: sse2\nflags: sse2 sse4_1\n",
        "flags: sse2\nflags:\n",
    ],
)
def test_capable_or_unknown_cpu_flags_preserve_backend(host, cpuinfo):
    host.cpuinfo.read_text = lambda *args, **kwargs: cpuinfo
    policy.pytest_configure(host.config)
    assert host.backend.enabled is True
    assert host.cleanups == []
    assert policy.pytest_report_header(host.config) is None


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("CPU flags are inaccessible"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid CPU flags"),
    ],
)
def test_cpuinfo_read_errors_preserve_backend(host, error):
    def unavailable(*args, **kwargs):
        raise error

    host.cpuinfo.read_text = unavailable
    policy.pytest_configure(host.config)
    assert host.backend.enabled is True
    assert host.cleanups == []
    assert policy.pytest_report_header(host.config) is None
