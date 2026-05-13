import argparse
import json
import re
from pathlib import Path


SCF_CRITERION_RE = re.compile(r"^\s*scf\.criterion\s+([0-9Ee+\-.]+)")
SCF_MAXITER_RE = re.compile(r"^\s*scf\.maxIter\s+(\d+)")
SYSTEM_NAME_RE = re.compile(r"^\s*System\.Name\s+(.+?)\s*$")
SCF_LINE_RE = re.compile(
    r"^\s*SCF=\s*(\d+)\s+NormRD=\s*([0-9Ee+\-.]+)\s+Uele=\s*([0-9Ee+\-.]+)"
)
ENERGY_RE = re.compile(r"^\s*(Utot|Uele|Enpy)\.\s+([0-9Ee+\-.]+)")


def parse_openmx_output(path: Path, input_dir: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")

    criterion = None
    max_iter = None
    system_name = None
    scf_steps = []
    energies = {}

    for line in text.splitlines():
        if criterion is None:
            match = SCF_CRITERION_RE.match(line)
            if match:
                criterion = float(match.group(1))

        if max_iter is None:
            match = SCF_MAXITER_RE.match(line)
            if match:
                max_iter = int(match.group(1))

        if system_name is None:
            match = SYSTEM_NAME_RE.match(line)
            if match:
                system_name = match.group(1).strip()

        match = SCF_LINE_RE.match(line)
        if match:
            scf_steps.append(
                {
                    "step": int(match.group(1)),
                    "normrd": float(match.group(2)),
                    "uele": float(match.group(3)),
                }
            )
            continue

        match = ENERGY_RE.match(line)
        if match:
            energies[match.group(1).lower()] = float(match.group(2))

    last_scf = scf_steps[-1] if scf_steps else None
    has_total_energy = "utot" in energies or "enpy" in energies

    if last_scf is None:
        status = "no_scf_data"
        converged = False
    elif criterion is None:
        status = "unknown_criterion"
        converged = False
    elif last_scf["normrd"] <= criterion:
        status = "converged"
        converged = True
    elif max_iter is not None and last_scf["step"] >= max_iter:
        status = "not_converged"
        converged = False
    elif has_total_energy:
        status = "finished_without_convergence"
        converged = False
    else:
        status = "incomplete"
        converged = False

    return {
        "file": str(path.resolve()),
        "relative_file": str(path.relative_to(input_dir)),
        "system_name": system_name,
        "status": status,
        "converged": converged,
        "scf_criterion": criterion,
        "scf_max_iter": max_iter,
        "last_scf_step": None if last_scf is None else last_scf["step"],
        "last_normrd": None if last_scf is None else last_scf["normrd"],
        "last_uele": None if last_scf is None else last_scf["uele"],
        "energies": {
            "uele": energies.get("uele"),
            "utot": energies.get("utot"),
            "enpy": energies.get("enpy"),
        },
    }


def collect_outputs(input_dir: Path) -> list[Path]:
    return sorted(path.relative_to(input_dir) for path in input_dir.rglob("*.out"))


def build_report(input_dir: Path) -> dict:
    relative_paths = collect_outputs(input_dir)
    calculations = [
        parse_openmx_output(input_dir / relative_path, input_dir)
        for relative_path in relative_paths
    ]

    summary = {
        "total": len(calculations),
        "converged": sum(item["converged"] for item in calculations),
        "not_converged": sum(item["status"] == "not_converged" for item in calculations),
        "incomplete": sum(item["status"] == "incomplete" for item in calculations),
        "other": sum(
            item["status"] not in {"converged", "not_converged", "incomplete"}
            for item in calculations
        ),
    }

    return {
        "input_dir": str(input_dir.resolve()),
        "summary": summary,
        "calculations": calculations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan OpenMX *.out files and export convergence/energy status as JSON."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory to recursively scan for OpenMX *.out files.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_json = args.output_json.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    report = build_report(input_dir)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scanned {report['summary']['total']} OpenMX output files.")
    print(f"Saved report to {output_json}")


if __name__ == "__main__":
    main()
