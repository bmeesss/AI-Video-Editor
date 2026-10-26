"""Phase 4 tests: OpenCode skills frontmatter/discovery, scripts (diagnose, verify_project, install.ps1), and CLI E2E."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_video_editor.cli import main as cli_main
from ai_video_editor.safety import capture_source_snapshot, verify_source_untouched

from .synthetic_media import create_synthetic_gaming_video


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SKILLS = [
    "analyze-gaming-video",
    "generate-timeline-project",
    "verify-and-diagnose",
    "install-ai-video-editor",
]


def _parse_yaml_frontmatter(md_text: str) -> dict[str, str]:
    assert md_text.startswith("---\n"), "SKILL.md must begin with YAML frontmatter delimiter '---'"
    end_idx = md_text.find("\n---\n", 4)
    assert end_idx != -1, "SKILL.md is missing closing YAML frontmatter delimiter '---'"
    fm_block = md_text[4:end_idx]
    parsed: dict[str, str] = {}
    for line in fm_block.splitlines():
        if not line.strip() or line.startswith(" ") or line.startswith("\t"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            parsed[k.strip()] = v.strip()
    return parsed


def test_opencode_skills_discovery_and_frontmatter_compliance() -> None:
    skills_root = REPO_ROOT / ".opencode" / "skills"
    assert skills_root.is_dir(), ".opencode/skills directory must exist"

    name_regex = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

    for skill_name in EXPECTED_SKILLS:
        skill_dir = skills_root / skill_name
        skill_file = skill_dir / "SKILL.md"
        assert skill_dir.is_dir(), f"Missing skill directory: {skill_dir}"
        assert skill_file.is_file(), f"Missing SKILL.md in {skill_dir}"

        content = skill_file.read_text(encoding="utf-8")
        fm = _parse_yaml_frontmatter(content)

        assert "name" in fm, f"{skill_file} missing 'name' in frontmatter"
        assert fm["name"] == skill_name, f"Frontmatter name '{fm['name']}' must match folder '{skill_name}'"
        assert 1 <= len(fm["name"]) <= 64
        assert name_regex.match(fm["name"]), f"Invalid skill name format: {fm['name']}"

        assert "description" in fm, f"{skill_file} missing 'description' in frontmatter"
        assert 1 <= len(fm["description"]) <= 1024

        assert fm.get("license") == "MIT"
        assert fm.get("compatibility") == "opencode"


def test_windows_powershell_installer_script_exists_and_is_well_formed() -> None:
    install_ps1 = REPO_ROOT / "scripts" / "install.ps1"
    assert install_ps1.is_file()
    text = install_ps1.read_text(encoding="utf-8")
    assert "PreDownloadWhisperModel" in text
    assert "-m venv" in text
    assert "scripts\\diagnose.py" in text


def test_diagnose_script_and_cli_subcommand() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "diagnose.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    assert data["all_required_ok"] is True
    assert data["recommended_whisper_model"] in ("base", "small")


def test_verify_project_script_generates_and_validates_phase0_minimal_sample(tmp_path: Path) -> None:
    sample_base = tmp_path / "phase0_sample_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_project.py"),
            "--generate-minimal-sample",
            str(sample_base),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["verification"]["is_valid"] is True
    assert payload["verification"]["dtd_valid"] is True
    assert payload["verification"]["kdenlive_semantics_valid"] is True
    assert Path(payload["kdenlive_project"]).is_file()
    assert Path(payload["kdenlive_srt"]).is_file()
    assert Path(payload["standalone_srt"]).is_file()


def test_end_to_end_cli_workflow_preserves_source_and_never_overwrites(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_video = create_synthetic_gaming_video(
        tmp_path / "source_dir" / "Jürgen's Match #99 (1080p).mp4",
        duration_seconds=20,
        fps=30,
        dual_audio_tracks=True,
    )
    snap_initial = capture_source_snapshot(source_video)
    out_base = tmp_path / "edited_projects"

    # Run 1: Full 'edit' command in default review-layers mode
    rc1 = cli_main(
        [
            "edit",
            "--input",
            str(source_video),
            "--title",
            "Jürgen's Match #99: Boss Victory!",
            "--output-dir",
            str(out_base),
            "--timeline-mode",
            "review-layers",
            "--preserve-intro",
            "4.5",
            "--min-silence",
            "2.5",
        ]
    )
    assert rc1 == 0
    out1 = json.loads(capsys.readouterr().out)
    assert out1["status"] == "ok"
    assert out1["verification_is_valid"] is True

    # Run 2: Identical title & output-dir -> must create a second unique directory and never overwrite Run 1
    rc2 = cli_main(
        [
            "edit",
            "--input",
            str(source_video),
            "--title",
            "Jürgen's Match #99: Boss Victory!",
            "--output-dir",
            str(out_base),
            "--timeline-mode",
            "ripple",
            "--preserve-intro",
            "4.5",
            "--min-silence",
            "2.5",
        ]
    )
    assert rc2 == 0
    out2 = json.loads(capsys.readouterr().out)
    assert out2["status"] == "ok"
    assert out1["job_dir"] != out2["job_dir"]
    assert Path(out1["kdenlive_project"]).is_file()
    assert Path(out2["kdenlive_project"]).is_file()

    # Run 3: Verify command on the generated project
    rc_ver = cli_main(["verify", out1["kdenlive_project"]])
    assert rc_ver == 0
    ver_out = json.loads(capsys.readouterr().out)
    assert ver_out["is_valid"] is True

    # Source video must remain 100% bit-identical
    assert verify_source_untouched(snap_initial) is True
