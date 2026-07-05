# Tabelas para relatorio

## Ranking global composto

| Rank | Eixo | Metodo | Score | SSIM | PSNR (dB) | MSE | HFEN | ISNR (dB) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `axial` | `lanczos` | 0.9761 | 0.6817 | 20.81 | 0.008328 | 0.1518 | 0.641 |
| 2 | `axial` | `bicubic` | 0.9694 | 0.6838 | 20.77 | 0.008410 | 0.1532 | 0.599 |
| 3 | `coronal` | `lanczos` | 0.8656 | 0.6685 | 20.76 | 0.008393 | 0.1595 | 0.556 |
| 4 | `coronal` | `bicubic` | 0.8557 | 0.6702 | 20.74 | 0.008439 | 0.1606 | 0.532 |
| 5 | `axial` | `nearest` | 0.6182 | 0.6635 | 20.19 | 0.009600 | 0.1758 | 0.018 |
| 6 | `sagittal` | `lanczos` | 0.6106 | 0.6565 | 20.19 | 0.009602 | 0.1602 | 0.671 |
| 7 | `sagittal` | `bicubic` | 0.5852 | 0.6567 | 20.14 | 0.009711 | 0.1626 | 0.622 |
| 8 | `coronal` | `nearest` | 0.5780 | 0.6542 | 20.26 | 0.009435 | 0.1823 | 0.048 |
| 9 | `axial` | `bilinear` | 0.4775 | 0.6556 | 20.17 | 0.009652 | 0.2410 | 0.000 |
| 10 | `coronal` | `bilinear` | 0.4190 | 0.6442 | 20.21 | 0.009539 | 0.2459 | 0.000 |
| 11 | `sagittal` | `nearest` | 0.2405 | 0.6406 | 19.58 | 0.011050 | 0.1877 | 0.058 |
| 12 | `sagittal` | `bilinear` | 0.0008 | 0.6164 | 19.52 | 0.011208 | 0.2599 | 0.000 |

## Ranking global por metodo

| Rank | Metodo | Score | Eixos | Fatias | SSIM | PSNR (dB) | MSE | HFEN | ISNR (dB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `lanczos` | 0.9447 | 3 | 93 | 0.6689 | 20.59 | 0.008774 | 0.1572 | 0.622 |
| 2 | `bicubic` | 0.9173 | 3 | 93 | 0.6702 | 20.55 | 0.008853 | 0.1588 | 0.584 |
| 3 | `nearest` | 0.3286 | 3 | 93 | 0.6528 | 20.01 | 0.010028 | 0.1819 | 0.041 |
| 4 | `bilinear` | 0.0258 | 3 | 93 | 0.6388 | 19.97 | 0.010133 | 0.2489 | 0.000 |

## Tamanho medio das imagens

| Eixo | Fatias | Escala | HR (HxW) | LR (HxW) | HR pixels | LR pixels | Razao HR/LR |
|---|---:|---:|---|---|---:|---:|---:|
| `axial` | 31 | 2.00 | 256x236 | 128x118 | 60416 | 15104 | 4.00 |
| `coronal` | 31 | 2.00 | 256x236 | 128x118 | 60416 | 15104 | 4.00 |
| `sagittal` | 31 | 2.00 | 256x256 | 128x128 | 65536 | 16384 | 4.00 |

## Tempos de processamento medios

| Eixo | Metodo | Degradacao (ms) | Upscaling (ms) | Metricas (ms) | Total por metodo (ms) |
|---|---|---:|---:|---:|---:|
| `axial` | `nearest` | 0.24 | 0.11 | 7.39 | 7.50 |
| `axial` | `bilinear` | 0.24 | 0.12 | 7.55 | 7.67 |
| `axial` | `bicubic` | 0.24 | 0.15 | 7.58 | 7.73 |
| `axial` | `lanczos` | 0.24 | 0.23 | 7.57 | 7.80 |
| `coronal` | `nearest` | 0.25 | 0.12 | 7.80 | 7.91 |
| `coronal` | `bilinear` | 0.25 | 0.14 | 7.92 | 8.06 |
| `coronal` | `lanczos` | 0.25 | 0.24 | 8.07 | 8.31 |
| `coronal` | `bicubic` | 0.25 | 0.17 | 8.29 | 8.46 |
| `sagittal` | `nearest` | 0.44 | 0.12 | 8.45 | 8.57 |
| `sagittal` | `lanczos` | 0.44 | 0.27 | 8.40 | 8.67 |
| `sagittal` | `bilinear` | 0.44 | 0.14 | 8.56 | 8.70 |
| `sagittal` | `bicubic` | 0.44 | 0.17 | 8.55 | 8.72 |