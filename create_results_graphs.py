"""Create summary graphs from the final aggregate CSV results."""

import argparse
from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt


def ler_csv(path: Path):
    """Read CSV rows from disk."""
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    """Parse input and output paths for graph generation."""
    parser = argparse.ArgumentParser(description="Create graphs from aggregate baseline results")
    parser.add_argument("--input-csv", default="results/analysis/final_comparison/all_axes_aggregate.csv")
    parser.add_argument("--output-dir", default="results/analysis/final_comparison/graficos")
    return parser.parse_args()


def para_float(valor, default=float("nan")):
    """Convert values to float and keep NaN for missing entries."""
    try:
        if valor is None or valor == "":
            return default
        return float(valor)
    except Exception:
        return default


def etiqueta_linha(row):
    """Build a compact label with axis and interpolation method."""
    return f"{row['slice_axis']} - {row['method']}"


def ordenar_por_metrica(rows, coluna, descendente=True):
    """Sort rows by one metric column."""
    return sorted(
        rows,
        key=lambda r: para_float(r.get(coluna)),
        reverse=descendente
    )


def coluna_tem_dados(rows, coluna):
    """Return True when at least one finite value exists for the column."""
    valores = [para_float(r.get(coluna)) for r in rows]
    return any(not math.isnan(v) for v in valores)


def grafico_barras(rows, coluna, titulo, ylabel, nome_ficheiro, output_dir: Path, descendente=True):
    """Render and save a bar chart for a single aggregate metric."""
    rows_ordenadas = ordenar_por_metrica(rows, coluna, descendente=descendente)
    labels = [etiqueta_linha(r) for r in rows_ordenadas]
    values = [para_float(r.get(coluna)) for r in rows_ordenadas]

    plt.figure(figsize=(12, 6), dpi=180)
    plt.bar(labels, values)
    plt.title(titulo)
    plt.xlabel("Orientação anatómica e método")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / nome_ficheiro)
    plt.close()


def grafico_ranking_global_ssim(rows, output_dir: Path):
    """Render and save a line chart following the global SSIM ranking rule."""
    rows_ordenadas = sorted(
        rows,
        key=lambda r: (
            -para_float(r.get("ssim_mean")),
            -para_float(r.get("psnr_mean")),
            para_float(r.get("mse_mean")),
            para_float(r.get("hfen_mean")),
        )
    )

    labels = [etiqueta_linha(r) for r in rows_ordenadas]
    values = [para_float(r.get("ssim_mean")) for r in rows_ordenadas]

    plt.figure(figsize=(12, 6), dpi=180)
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
    plt.xlabel("Ranking global")
    plt.ylabel("SSIM médio")
    plt.title("Ranking global com base no SSIM médio")
    plt.tight_layout()
    plt.savefig(output_dir / "ranking_global_ssim.png")
    plt.close()


def main():
    """Generate all default baseline comparison figures."""
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Não foi encontrado o ficheiro: {input_csv}\n"
            "Corre primeiro o script run_all_experiments.py."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = ler_csv(input_csv)

    # Quality-focused metrics where higher values are better.
    grafico_barras(
        rows,
        coluna="ssim_mean",
        titulo="Comparação do SSIM médio por orientação e método",
        ylabel="SSIM médio",
        nome_ficheiro="grafico_ssim_medio.png",
        output_dir=output_dir,
        descendente=True,
    )

    grafico_barras(
        rows,
        coluna="psnr_mean",
        titulo="Comparação do PSNR médio por orientação e método",
        ylabel="PSNR médio (dB)",
        nome_ficheiro="grafico_psnr_medio.png",
        output_dir=output_dir,
        descendente=True,
    )

    # Error-focused metrics where lower values are better.
    grafico_barras(
        rows,
        coluna="mse_mean",
        titulo="Comparação do MSE médio por orientação e método",
        ylabel="MSE médio",
        nome_ficheiro="grafico_mse_medio.png",
        output_dir=output_dir,
        descendente=False,
    )

    grafico_barras(
        rows,
        coluna="hfen_mean",
        titulo="Comparação do HFEN médio por orientação e método",
        ylabel="HFEN médio",
        nome_ficheiro="grafico_hfen_medio.png",
        output_dir=output_dir,
        descendente=False,
    )

    # Optional metrics are generated only when the source data contains them.
    if coluna_tem_dados(rows, "isnr_mean"):
        grafico_barras(
            rows,
            coluna="isnr_mean",
            titulo="Comparação do ISNR médio por orientação e método",
            ylabel="ISNR médio (dB)",
            nome_ficheiro="grafico_isnr_medio.png",
            output_dir=output_dir,
            descendente=True,
        )

    if coluna_tem_dados(rows, "total_method_time_ms_mean"):
        grafico_barras(
            rows,
            coluna="total_method_time_ms_mean",
            titulo="Tempo total médio por orientação e método",
            ylabel="Tempo médio por método (ms)",
            nome_ficheiro="grafico_tempo_total_medio.png",
            output_dir=output_dir,
            descendente=False,
        )

    grafico_ranking_global_ssim(rows, output_dir=output_dir)

    print("Gráficos gerados em:")
    print(output_dir)


if __name__ == "__main__":
    main()
