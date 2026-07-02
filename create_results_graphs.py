"""
Gera gráficos simples a partir do ficheiro:
    results/final_comparison/all_axes_aggregate.csv

Gráficos gerados:
    - grafico_ssim_medio.png
    - grafico_psnr_medio.png
    - grafico_mse_medio.png
    - grafico_hfen_medio.png
    - ranking_global_ssim.png

Todos os gráficos são guardados em:
    results/final_comparison/graficos/
"""

from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt


INPUT_CSV = Path("results/final_comparison/all_axes_aggregate.csv")
OUTPUT_DIR = Path("results/final_comparison/graficos")


def ler_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def para_float(valor, default=float("nan")):
    try:
        if valor is None or valor == "":
            return default
        return float(valor)
    except Exception:
        return default


def etiqueta_linha(row):
    return f"{row['slice_axis']} - {row['method']}"


def ordenar_por_metrica(rows, coluna, descendente=True):
    return sorted(
        rows,
        key=lambda r: para_float(r.get(coluna)),
        reverse=descendente
    )


def grafico_barras(rows, coluna, titulo, ylabel, nome_ficheiro, descendente=True):
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
    plt.savefig(OUTPUT_DIR / nome_ficheiro)
    plt.close()


def grafico_ranking_global_ssim(rows):
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
    plt.savefig(OUTPUT_DIR / "ranking_global_ssim.png")
    plt.close()


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Não foi encontrado o ficheiro: {INPUT_CSV}\n"
            "Corre primeiro o script run_all_experiments.py."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = ler_csv(INPUT_CSV)

    grafico_barras(
        rows,
        coluna="ssim_mean",
        titulo="Comparação do SSIM médio por orientação e método",
        ylabel="SSIM médio",
        nome_ficheiro="grafico_ssim_medio.png",
        descendente=True,
    )

    grafico_barras(
        rows,
        coluna="psnr_mean",
        titulo="Comparação do PSNR médio por orientação e método",
        ylabel="PSNR médio (dB)",
        nome_ficheiro="grafico_psnr_medio.png",
        descendente=True,
    )

    grafico_barras(
        rows,
        coluna="mse_mean",
        titulo="Comparação do MSE médio por orientação e método",
        ylabel="MSE médio",
        nome_ficheiro="grafico_mse_medio.png",
        descendente=False,
    )

    grafico_barras(
        rows,
        coluna="hfen_mean",
        titulo="Comparação do HFEN médio por orientação e método",
        ylabel="HFEN médio",
        nome_ficheiro="grafico_hfen_medio.png",
        descendente=False,
    )

    grafico_ranking_global_ssim(rows)

    print("Gráficos gerados em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
