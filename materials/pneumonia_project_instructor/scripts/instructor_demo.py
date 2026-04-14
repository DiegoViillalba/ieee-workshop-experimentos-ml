#!/usr/bin/env python
# scripts/instructor_demo.py
# Script de demo para el instructor: ejecuta 2 runs automáticamente
# (baseline + lr_high) y muestra la comparación de resultados al final.
#
# USO — ejecutar ANTES de la sesión para verificar que todo funciona:
#   uv run python scripts/instructor_demo.py
#
# USO — ejecutar DURANTE la sesión para demo en vivo:
#   uv run python scripts/instructor_demo.py --live
#   (con --live muestra más output intermedio)

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_cmd(cmd: list[str], label: str) -> bool:
    """Ejecuta un comando y devuelve True si tuvo éxito."""
    print(f"\n{'─'*60}")
    print(f"▶  {label}")
    print(f"   {' '.join(cmd)}")
    print('─'*60)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n✗ FALLÓ: {label}")
        return False
    print(f"\n✓ OK: {label}")
    return True


def load_metrics(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def print_comparison(run1_metrics: dict, run2_metrics: dict,
                     run1_label: str, run2_label: str) -> None:
    print("\n" + "═"*60)
    print("        Comparación de resultados en val set")
    print("═"*60)
    fmt = "{:<30s} {:>12s} {:>12s}"
    print(fmt.format("Métrica", run1_label[:12], run2_label[:12]))
    print("─"*60)

    # Extraer las últimas métricas de val del historial
    def last(d: dict, key: str) -> str:
        vals = d.get(key, [])
        return f"{vals[-1]:.4f}" if vals else "—"

    rows = [
        ("val/recall",   "val/recall"),
        ("val/auc",      "val/auc"),
        ("train/loss",   "train/loss"),
    ]
    for label, key in rows:
        print(fmt.format(label, last(run1_metrics, key), last(run2_metrics, key)))
    print("═"*60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo instructor: ejecuta 2 runs y compara resultados."
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Modo demo en vivo: pausa entre runs para explicar cada paso."
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Desactivar W&B (útil para prueba rápida sin conexión)."
    )
    parser.add_argument(
        "--config1", default="configs/baseline.yaml",
        help="Config del Run 1 (default: baseline)."
    )
    parser.add_argument(
        "--config2", default="configs/exercise_lr_high.yaml",
        help="Config del Run 2 (default: exercise_lr_high)."
    )
    args = parser.parse_args()

    base_cmd = [sys.executable, "scripts/train.py"]
    if args.no_wandb:
        base_cmd.append("--no-wandb")

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Demo instructor — Día 2 Workshop ML                ║")
    print("║   2 runs: baseline vs. lr_high                       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\nRun 1: {args.config1}")
    print(f"Run 2: {args.config2}")

    if args.live:
        input("\n[Presiona Enter para iniciar Run 1 (baseline)...]")

    # ── Run 1 ─────────────────────────────────────────────────────────────────
    ok1 = run_cmd(base_cmd + ["--config", args.config1], f"Run 1 — {args.config1}")
    if not ok1:
        print("Abortando demo.")
        sys.exit(1)

    # Renombrar el checkpoint y el historial del Run 1 para no pisarlo
    Path("outputs/checkpoints/best_model.pt").rename(
        "outputs/checkpoints/run1_best_model.pt"
    )
    Path("outputs/metrics/train_history.json").rename(
        "outputs/metrics/run1_train_history.json"
    )

    if args.live:
        print("\n" + "─"*60)
        print("Checkpoint Run 1 guardado en: outputs/checkpoints/run1_best_model.pt")
        print("Abre el dashboard de W&B y muestra las curvas val/recall.")
        input("\n[Presiona Enter para iniciar Run 2 (lr_high)...]")

    # ── Run 2 ─────────────────────────────────────────────────────────────────
    ok2 = run_cmd(base_cmd + ["--config", args.config2], f"Run 2 — {args.config2}")
    if not ok2:
        print("Run 2 falló.")
        sys.exit(1)

    Path("outputs/checkpoints/best_model.pt").rename(
        "outputs/checkpoints/run2_best_model.pt"
    )
    Path("outputs/metrics/train_history.json").rename(
        "outputs/metrics/run2_train_history.json"
    )

    # ── Comparación ───────────────────────────────────────────────────────────
    r1 = load_metrics("outputs/metrics/run1_train_history.json")
    r2 = load_metrics("outputs/metrics/run2_train_history.json")

    if r1 and r2:
        label1 = Path(args.config1).stem
        label2 = Path(args.config2).stem
        print_comparison(r1, r2, label1, label2)

    print("\n✓ Demo completada.")
    print("  Run 1 checkpoint : outputs/checkpoints/run1_best_model.pt")
    print("  Run 2 checkpoint : outputs/checkpoints/run2_best_model.pt")
    print("  Compara los runs en: https://wandb.ai")
    print()
    print("  Para evaluar Run 1:")
    print(f"    uv run python scripts/evaluate.py \\")
    print(f"      --config {args.config1} \\")
    print(f"      --checkpoint outputs/checkpoints/run1_best_model.pt")


if __name__ == "__main__":
    main()
