"""
End-to-End Demo for StackGate:
Creates realistic baseline and candidate profile runs with an injected regression,
runs comparison with function-level attribution, and performs A/A calibration.
"""

from pathlib import Path
import sys
import stackgate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    print("=" * 70)
    print("STACKGATE END-TO-END DEMO")
    print("=" * 70)

    base_dir = Path("examples/data/baseline")
    cand_dir = Path("examples/data/candidate")
    base_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate 8 baseline runs (normal variation around 1000 samples)
    import random
    rng = random.Random(42)

    for i in range(1, 9):
        p_base = (
            f"main;handle_request;parse_json {int(rng.gauss(300, 10))}\n"
            f"main;handle_request;validate_token {int(rng.gauss(100, 5))}\n"
            f"main;handle_request;db_query {int(rng.gauss(500, 15))}\n"
            f"main;handle_request;render_response {int(rng.gauss(100, 5))}\n"
        )
        (base_dir / f"run_{i:02d}.folded").write_text(p_base, encoding="utf-8")

    # 2. Generate 8 candidate runs with an injected regression in parse_json (+200 samples = +20% total)
    for i in range(1, 9):
        p_cand = (
            f"main;handle_request;parse_json {int(rng.gauss(500, 12))}\n"  # Regressed!
            f"main;handle_request;validate_token {int(rng.gauss(100, 5))}\n"
            f"main;handle_request;db_query {int(rng.gauss(500, 15))}\n"
            f"main;handle_request;render_response {int(rng.gauss(100, 5))}\n"
            f"main;handle_request;new_telemetry_hook {int(rng.gauss(20, 2))}\n"  # New path!
        )
        (cand_dir / f"run_{i:02d}.folded").write_text(p_cand, encoding="utf-8")

    print("\n1. Running noise-aware comparison...")
    result = stackgate.compare(
        baseline=base_dir,
        candidate=cand_dir,
        threshold=2.0,
        alpha=0.05,
        metric="self",
    )

    print("\n--- Terminal Output ---")
    print(result.to_terminal(use_color=True))

    print("\n--- GitHub PR Markdown Comment ---")
    print(result.to_markdown())

    print("\n2. Running empirical A/A calibration on baseline runs...")
    calib = stackgate.calibrate(base_dir, alpha=0.05, num_splits=50)
    print(calib.summary())


if __name__ == "__main__":
    main()
