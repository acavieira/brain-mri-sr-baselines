# Análise Visual e Quantitativa Avançada

Este documento complementa o resumo final da experiência baseline de super-resolução MRI.

A análise inclui gráficos que procuram responder a perguntas mais úteis do que apenas “qual é o maior valor médio?”:

- Que método preserva melhor a estrutura?
- Que método preserva melhor o detalhe fino?
- O desempenho é estável entre fatias?
- Há diferenças relevantes entre orientações anatómicas?
- Que método ganha mais vezes fatia a fatia?
- Qual é o melhor compromisso entre SSIM, PSNR, MSE e HFEN?

## Melhor compromisso global

O melhor compromisso global pelo score composto foi `lanczos` na orientação `coronal`.

- Score composto: `0.9642`
- SSIM médio: `0.5656`
- PSNR médio: `18.81`
- MSE médio: `0.013148`
- HFEN médio: `0.4635`

## Gráficos gerados

| Ficheiro | O que mostra | Como interpretar |
|---|---|---|
| `01_matriz_ssim_medio.png` | SSIM médio por orientação e método | Maior é melhor; mostra preservação estrutural |
| `02_matriz_psnr_medio.png` | PSNR médio por orientação e método | Maior é melhor; mostra qualidade global baseada no erro |
| `03_matriz_mse_medio.png` | MSE médio por orientação e método | Menor é melhor; mostra erro pixel a pixel |
| `04_matriz_hfen_medio.png` | HFEN médio por orientação e método | Menor é melhor; mostra erro em detalhe fino |
| `05_tradeoff_ssim_hfen.png` | Relação entre SSIM e HFEN | Idealmente, pontos mais acima e mais à esquerda são melhores |
| `06_score_composto.png` | Ranking combinado | Resume o compromisso entre várias métricas |
| `07_distribuicao_ssim_por_fatia.png` | Distribuição do SSIM nas fatias | Mostra se o desempenho é estável ou variável |
| `08_distribuicao_hfen_por_fatia.png` | Distribuição do HFEN nas fatias | Mostra estabilidade na preservação de detalhe fino |
| `09_estabilidade_ssim_std.png` | Desvio padrão do SSIM | Menor indica comportamento mais estável entre fatias |
| `10_ganho_ssim_vs_bilinear.png` | Ganho percentual de SSIM face ao bilinear | Mostra melhoria relativa face a uma baseline simples |
| `11_reducao_mse_vs_bilinear.png` | Redução percentual de MSE face ao bilinear | Mostra quanto erro foi reduzido face ao bilinear |
| `12_vitorias_por_fatia_ssim.png` | Número de fatias em que cada método teve melhor SSIM | Mostra consistência fatia a fatia |
| `13_vitorias_por_fatia_hfen.png` | Número de fatias em que cada método teve melhor HFEN | Mostra consistência na preservação de detalhe fino |

## Contagem de vitórias por SSIM

### Sagital
- `nearest`: 0 fatias
- `bilinear`: 0 fatias
- `bicubic`: 0 fatias
- `lanczos`: 9 fatias

### Coronal
- `nearest`: 0 fatias
- `bilinear`: 0 fatias
- `bicubic`: 0 fatias
- `lanczos`: 9 fatias

### Axial
- `nearest`: 0 fatias
- `bilinear`: 0 fatias
- `bicubic`: 0 fatias
- `lanczos`: 9 fatias

## Contagem de vitórias por HFEN

### Sagital
- `nearest`: 0 fatias
- `bilinear`: 0 fatias
- `bicubic`: 0 fatias
- `lanczos`: 9 fatias

### Coronal
- `nearest`: 0 fatias
- `bilinear`: 0 fatias
- `bicubic`: 0 fatias
- `lanczos`: 9 fatias

### Axial
- `nearest`: 0 fatias
- `bilinear`: 0 fatias
- `bicubic`: 0 fatias
- `lanczos`: 9 fatias

## Nota metodológica

O score composto é apenas uma ferramenta exploratória para resumir várias métricas num único ranking.
Não substitui a análise individual das métricas nem a inspeção dos mapas de erro.

Em imagem médica, um método não deve ser considerado superior apenas por ter melhor PSNR ou MSE.
É importante avaliar simultaneamente a similaridade estrutural, o erro em detalhe fino, a estabilidade entre fatias e a interpretação visual.
