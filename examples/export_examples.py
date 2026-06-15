"""Export raw and optimized OptCPV examples."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optcpv import draw_artifact, draw_optimized_artifact
from optcpv.examples import EXAMPLES


def main() -> None:
    output_dir = Path("generated")
    output_dir.mkdir(exist_ok=True)
    for name, factory in EXAMPLES.items():
        raw = draw_artifact(factory())
        optimized = draw_optimized_artifact(factory())
        (output_dir / f"{name}.raw.svg").write_text(raw.svg, encoding="utf-8")
        (output_dir / f"{name}.optimized.svg").write_text(optimized.svg, encoding="utf-8")
        (output_dir / f"{name}.artifact.json").write_text(
            json.dumps(_artifact_json(optimized), indent=2),
            encoding="utf-8",
        )
        (output_dir / f"{name}.critic.json").write_text(
            json.dumps(optimized.critic_report or {}, indent=2),
            encoding="utf-8",
        )
        raw_score = raw.critic_report["score"] if raw.critic_report else 0
        optimized_score = optimized.critic_report["score"] if optimized.critic_report else 0
        print(f"{name}: raw score {raw_score:.2f} -> optimized score {optimized_score:.2f}")


def _artifact_json(artifact) -> dict:
    return {
        "components": artifact.components,
        "nets": artifact.nets,
        "labels": artifact.labels,
        "viewbox": artifact.viewbox,
        "critic_report": artifact.critic_report,
        "cv_report": artifact.cv_report,
        "optimization_log": artifact.optimization_log,
        "warnings": artifact.warnings,
    }


if __name__ == "__main__":
    main()
