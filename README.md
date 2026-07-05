# MRI Super-Resolution Baseline Experiment

Este repositório contém o pipeline da minha tese de mestrado para super-resolução em imagem médica (MRI cerebral em NIfTI), incluindo:

- baseline clássica de interpolação;
- geração de tabelas para relatório/tese;
- estrutura inicial de diffusion model com U-Net condicional treinada de raiz.

O pipeline parte de volumes MRI, extrai fatias 2D, cria versões LR sintéticas controladas e compara reconstruções SR com a referência HR.

A baseline permite perceber, de forma quantitativa e visual, até que ponto métodos como nearest-neighbour, bilinear, bicubic e Lanczos conseguem aproximar a imagem original depois de uma degradação artificial. Esta análise será útil para comparar, numa fase posterior, métodos mais avançados como CNNs, GANs, transformers ou diffusion models.

---

## Objetivo da experiência

O pipeline experimental é:

```text
Volume MRI 3D em formato NIfTI
        ↓
Extração de fatias 2D em diferentes orientações anatómicas
        ↓
Normalização robusta das intensidades
        ↓
Synthetic blur (Gaussiano)
        ↓
Downsampling (HR -> LR)
        ↓
Ruído opcional na LR
        ↓
Upscaling (métodos clássicos ou diffusion U-Net)
        ↓
Comparação com a fatia original high-resolution
        ↓
Cálculo de métricas quantitativas e tempos
        ↓
Geração de mapas de erro, tabelas e comparação final
```

A imagem original é tratada como **HR — High Resolution**.

A imagem degradada artificialmente é tratada como **LR — Low Resolution**.

A imagem reconstruída depois do upscaling é tratada como **SR — Super-Resolved image**.

---

## Orientações anatómicas avaliadas

O projeto não avalia apenas uma fatia sagital. A experiência foi preparada para correr automaticamente nas três principais orientações anatómicas:

| Orientação | Descrição |
|---|---|
| `sagittal` | Corte lateral do cérebro |
| `coronal` | Corte frontal |
| `axial` | Corte horizontal/transversal |

Para cada orientação, são extraídas várias fatias próximas da zona central do volume. Cada orientação gera a sua própria pasta de resultados.

Exemplo:

```text
results/
├── baseline_methods/
│   ├── scale2_sagittal/
│   ├── scale2_coronal/
│   └── scale2_axial/
├── analysis/
│   └── final_comparison/
└── diffusion/
    ├── training/
    ├── evaluation/
    └── diagrams/
```

Esta organização separa claramente:

- resultados da baseline de métodos clássicos;
- tabelas e análises finais;
- artefactos de treino/avaliação de difusão.

---

## Motivação

Em super-resolução médica, não basta aumentar o tamanho da imagem. É necessário avaliar se a reconstrução preserva estruturas anatómicas, contrastes locais, contornos e detalhe fino.

No caso da MRI cerebral, uma imagem pode parecer visualmente aceitável, mas ainda assim perder informação estrutural importante. Por isso, esta experiência junta:

- métricas de erro pixel-a-pixel;
- métricas de similaridade estrutural;
- métricas de preservação de detalhe;
- mapas de erro;
- mapas locais de SSIM;
- análise por orientação anatómica;
- comparação agregada entre métodos.

Esta organização permite transformar uma comparação visual simples numa análise mais clara, reprodutível e adequada ao desenvolvimento da tese.

---

## Estrutura do projeto

```text
mri-sr-experiment/
│
├── README.md
├── requirements.txt
├── run_experiment.py
├── run_all_experiments.py
├── build_thesis_tables.py
├── run_diffusion_training.py
├── run_diffusion_inference.py
├── create_pipeline_figure.py
├── create_diffusion_diagram.py
│
├── data/
│   └── sub-0_ses-1_T1w.nii
│
├── results/
│   ├── baseline_methods/
│   ├── analysis/
│   └── diffusion/
│
└── src/
    └── mri_sr/
        ├── __init__.py
        ├── config.py
        ├── io.py
        ├── preprocessing.py
        ├── degradation.py
        ├── upscaling.py
        ├── metrics.py
        ├── reports.py
        ├── experiment.py
        ├── thesis_tables.py
        └── diffusion/
            ├── config.py
            ├── dataset.py
            ├── model.py
            ├── scheduler.py
            ├── train.py
            └── inference.py
```

---

## Organização dos ficheiros

### `run_experiment.py`

Corre uma experiência específica, por exemplo apenas no eixo sagital.

É útil para testar rapidamente uma configuração.

---

### `run_all_experiments.py`

Corre a experiência completa nas três orientações anatómicas:

```text
sagittal
coronal
axial
```

Para cada orientação, cria uma pasta separada em:

```text
results/baseline_methods/scale{scale}_{axis}/
```

No final, cria os agregados em:

```text
results/analysis/final_comparison/
```

com:

```text
all_axes_aggregate.csv
all_axes_by_slice.csv
final_results_summary.md
tables/*.csv
```

O ficheiro `all_axes_aggregate.csv` junta os resultados agregados de todas as orientações.

O ficheiro `final_results_summary.md` apresenta uma comparação em formato legível, organizada por orientação e método.

---

### `build_thesis_tables.py`

Gera tabelas finais para o relatório/tese a partir de `results/analysis/final_comparison/`.

Principais saídas:

- ranking composto por eixo+método;
- ranking global por método (agregado entre eixos);
- tabela de tempos de processamento;
- tabela de tamanhos HR/LR;
- template CSV para comparação de PSNR com papers.

---

### `run_diffusion_training.py` e `run_diffusion_inference.py`

Scripts de treino e avaliação da diffusion U-Net condicional treinada de raiz.

- `run_diffusion_training.py`: treino DDPM com U-Net condicional (sem pretrained).
- `run_diffusion_inference.py`: carrega checkpoint e avalia no mesmo formato de métricas/tabelas da baseline.

---

### `create_pipeline_figure.py`

Gera uma figura esquemática do pipeline experimental.

A figura representa as etapas com a ordem correta do *synthetic blur*:

```text
NIfTI volume
    ↓
slice extraction
    ↓
preprocessing
    ↓
synthetic blur
    ↓
downsampling (HR -> LR)
    ↓
optional LR noise
    ↓
upscaling methods
    ↓
metrics and error maps
    ↓
final comparison
```

Saídas por omissão:

```text
results/analysis/final_comparison/diagrams/pipeline_baseline_en.png
results/analysis/final_comparison/diagrams/pipeline_baseline_pt.png
```

---

### `create_diffusion_diagram.py`

Gera um diagrama do core do modelo de difusão (treino + inferência) dentro da pasta de difusão.

Saída por omissão:

```text
results/diffusion/diagrams/diffusion_model_pipeline.png
```

---

### `config.py`

Contém a classe `ExperimentConfig`.

Esta classe agrupa todos os parâmetros da experiência, evitando passar muitos argumentos soltos entre funções.

Parâmetros principais:

| Parâmetro | Descrição |
|---|---|
| `input_path` | Caminho para o ficheiro `.nii` ou `.nii.gz` |
| `output_dir` | Pasta onde os resultados serão guardados |
| `slice_axis` | Orientação anatómica: `sagittal`, `coronal` ou `axial` |
| `slice_index` | Índice central da fatia; `-1` usa a fatia central |
| `num_slices` | Número de fatias vizinhas a testar |
| `scale` | Fator de downscale/upscale |
| `blur_sigma` | Intensidade do Gaussian blur antes do downsampling |
| `noise_sigma` | Ruído opcional aplicado à imagem LR |
| `force_square_size` | Redimensionamento opcional para tamanho fixo |
| `save_all_figures` | Guarda relatórios visuais para todas as fatias |

---

### `io.py`

Responsável pela leitura e extração de dados.

Funções principais:

- carregar o volume NIfTI;
- validar o formato do ficheiro;
- escolher os índices das fatias;
- extrair uma fatia 2D numa orientação específica.

A extração aplica uma correção simples de orientação visual para facilitar a interpretação das imagens geradas.

---

### `preprocessing.py`

Contém as funções de pré-processamento.

Os volumes NIfTI não têm necessariamente intensidades numa escala comum como uma imagem PNG. Por isso, é feita uma normalização robusta.

#### Normalização robusta

A função:

```python
robust_normalize_to_float01()
```

faz:

1. conversão para `float32`;
2. cálculo dos percentis 1 e 99;
3. corte de valores extremos;
4. normalização para o intervalo `[0,1]`.

Isto reduz o impacto de outliers e permite comparar imagens numa escala comum.

As métricas são calculadas em `float32`, não em `uint8`, para evitar perda de precisão.

#### Conversão para uint8

A função:

```python
float01_to_uint8()
```

é usada apenas para guardar imagens PNG.

Resumo:

```text
métricas      → float32 [0,1]
visualização  → uint8 [0,255]
```

---

### `degradation.py`

Cria a imagem LR sintética.

Processo:

```text
HR
 ↓
Gaussian blur opcional
 ↓
downsampling
 ↓
ruído opcional
 ↓
LR
```

O Gaussian blur antes do downsampling simula perda de detalhe fino antes da redução de resolução.

O downsampling é feito com `INTER_AREA`, uma opção adequada para reduzir imagens.

---

### `upscaling.py`

Contém os métodos clássicos de upscaling:

| Método | Descrição |
|---|---|
| `nearest` | Repete o valor do pixel mais próximo |
| `bilinear` | Interpola usando os vizinhos próximos |
| `bicubic` | Usa interpolação cúbica, geralmente mais suave |
| `lanczos` | Usa uma janela sinc, frequentemente preservando melhor detalhe |

Estes métodos são usados como baselines clássicos. Eles não aprendem a partir dos dados e não recuperam verdadeiramente informação anatómica perdida. Apenas estimam valores intermédios.

---

### `metrics.py`

Calcula as métricas quantitativas usadas para comparar HR e SR.

Todas as métricas são *full-reference*, porque comparam uma imagem reconstruída com uma referência conhecida.

A função principal é:

```python
compute_metrics()
```

---

## Métricas utilizadas

### MSE — Mean Squared Error

Mede o erro quadrático médio entre a imagem HR e a imagem SR.

```text
Menor é melhor.
```

Penaliza mais erros grandes porque eleva as diferenças ao quadrado.

É útil para medir erro global, mas pode não refletir bem a qualidade estrutural ou anatómica da imagem.

---

### MAE — Mean Absolute Error

Mede o erro absoluto médio entre HR e SR.

```text
Menor é melhor.
```

É mais simples de interpretar do que MSE e menos sensível a erros extremos.

---

### RMSE — Root Mean Squared Error

É a raiz quadrada do MSE.

```text
Menor é melhor.
```

Como as imagens estão normalizadas para `[0,1]`, o RMSE também fica nessa escala.

---

### NRMSE — Normalized Root Mean Squared Error

É o RMSE normalizado pela gama de intensidades da imagem.

```text
Menor é melhor.
```

Ajuda a comparar resultados em experiências com escalas de intensidade diferentes.

---

### PSNR — Peak Signal-to-Noise Ratio

Mede a relação entre o valor máximo possível da imagem e o erro de reconstrução.

```text
Maior é melhor.
```

É uma métrica clássica em super-resolução. Como está ligada ao MSE, pode favorecer imagens mais suaves, mesmo quando perdem detalhe fino.

---

### SSIM — Structural Similarity Index Measure

Mede a similaridade estrutural entre HR e SR.

```text
Maior é melhor.
```

Considera luminância, contraste e estrutura local.

É particularmente relevante em imagem médica, porque uma reconstrução pode ter erro pixel-a-pixel baixo mas ainda assim alterar estruturas anatómicas.

O projeto guarda também um mapa local de SSIM, que mostra em que zonas a similaridade estrutural é maior ou menor.

---

### Pearson correlation

Mede a correlação linear entre as intensidades da imagem HR e da imagem SR.

```text
Maior é melhor.
```

Ajuda a perceber se o padrão global de intensidades foi preservado.

---

### Gradient MSE

Compara os gradientes da imagem HR e SR.

```text
Menor é melhor.
```

Os gradientes estão associados a bordas, contornos e transições de intensidade.

Em MRI cerebral, esta métrica ajuda a perceber se a reconstrução suavizou ou alterou fronteiras anatómicas.

---

### HFEN — High Frequency Error Norm

Mede erro associado a detalhe fino e componentes de alta frequência.

```text
Menor é melhor.
```

É útil em super-resolução porque avalia se a reconstrução preserva detalhe fino, não apenas se aproxima valores médios.

Nesta implementação, é usada uma aproximação baseada em diferença de Gaussianas para isolar componentes de alta frequência.

---

### Diff%

Representa a diferença absoluta média em percentagem.

```text
Menor é melhor.
```

Como as imagens estão em `[0,1]`, é calculado como:

```text
MAE × 100
```

---

### ISSM — Information theoretic-based Statistic Similarity Measure

O ISSM é uma métrica adicional de similaridade entre imagens.

```text
Maior tende a indicar maior similaridade.
```

Ao contrário de métricas puramente baseadas em erro pixel-a-pixel, o ISSM combina informação estatística e informacional para avaliar similaridade de forma mais global.

Neste projeto, o ISSM é opcional e é calculado através da biblioteca `image-similarity-measures`.

Instalação:

```bash
pip install image-similarity-measures
```

Como as imagens MRI usadas aqui são grayscale e 2D, é necessário adicionar uma dimensão de canal:

```python
ref_3d = ref.astype(np.float32)[..., np.newaxis]
pred_3d = pred.astype(np.float32)[..., np.newaxis]
```

Isto transforma:

```text
(height, width)
```

em:

```text
(height, width, 1)
```

Se a biblioteca não estiver instalada ou se o cálculo falhar, o projeto continua a correr e o campo ISSM fica vazio no CSV.

---

## `reports.py`

Gera os relatórios visuais e os ficheiros CSV.

Cada relatório visual inclui:

```text
HR reference
LR degraded
SR / upscaled image
absolute error map
signed error map
local SSIM map
```

### HR reference

Fatia original normalizada, usada como referência de alta resolução.

### LR degraded

Versão artificialmente degradada por blur e downsampling.

### SR / upscaled image

Imagem reconstruída pelo método de upscaling.

### Absolute error map

Mostra:

```text
|HR - SR|
```

Indica onde a reconstrução mais se afastou da imagem original.

### Signed error map

Mostra:

```text
HR - SR
```

Ajuda a perceber se a reconstrução ficou mais clara ou mais escura em determinadas regiões.

### Local SSIM map

Mostra onde a estrutura foi melhor ou pior preservada.

---

## `experiment.py`

Orquestra a experiência completa.

Responsabilidades:

1. carregar o volume NIfTI;
2. escolher fatias;
3. extrair cada fatia;
4. preparar a referência HR;
5. criar a imagem LR;
6. aplicar cada método de upscaling;
7. calcular métricas;
8. guardar relatórios visuais;
9. guardar CSVs;
10. gerar ranking agregado.

---

## Instalação

Forma mais simples:

```bash
bash bootstrap_env.sh
```

Recomendado no macOS/Linux: usa Python 3.11 ou 3.12 para criar o ambiente virtual. Algumas builds de Python 3.13 empacotadas com conda/miniconda podem travar durante o bootstrap do `venv`/`ensurepip`.

Criar ambiente virtual:

```bash
python3.11 -m venv .venv
```

Se não tiveres `python3.11`, uma alternativa estável é usar conda:

```bash
conda create -n mri-sr python=3.11
conda activate mri-sr
```

Se criares `.venv`, ativa assim no Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se criares `.venv`, ativa assim no Mac/Linux:

```bash
source .venv/bin/activate
```

Se optares por conda, ativa o ambiente com:

```bash
conda activate mri-sr
```

Instalar dependências:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conteúdo recomendado do `requirements.txt`:

```txt
numpy
opencv-python
matplotlib
nibabel
image-similarity-measures
```

Para a parte de diffusion, instala também o PyTorch (com a wheel adequada ao teu sistema):

```bash
python -m pip install torch torchvision
```

Se o comando `python -m venv .venv` ficar bloqueado, é quase sempre um problema do Python/base environment e não do projeto. Nesse caso, cria o ambiente com outra versão do Python e repete a instalação acima.

---

## Dados de entrada

O ficheiro NIfTI deve ser colocado na pasta `data/`.

Exemplo:

```text
data/sub-0_ses-1_T1w.nii
```

---

## Como correr uma experiência simples

```bash
python run_experiment.py
```

---

## Como correr todas as orientações anatómicas

```bash
python run_all_experiments.py
```

Este comando cria resultados separados para:

```text
results/baseline_methods/scale2_sagittal/
results/baseline_methods/scale2_coronal/
results/baseline_methods/scale2_axial/
```

E depois cria uma comparação final:

```text
results/analysis/final_comparison/
├── all_axes_aggregate.csv
├── all_axes_by_slice.csv
├── final_results_summary.md
└── tables/
```

---

## Gerar tabelas para relatório/tese

```bash
python build_thesis_tables.py \
  --base-results-dir results/baseline_methods \
  --analysis-dir results/analysis/final_comparison \
  --scale 2
```

Saídas:

```text
results/analysis/final_comparison/tables/
├── ranking_global_composto.csv
├── ranking_global_metodos.csv
├── tamanho_imagens.csv
├── tempos_processamento.csv
├── psnr_vs_literatura_template.csv
└── tabelas_relatorio.md
```

---

## Atualizar o diagrama do pipeline

```bash
python create_pipeline_figure.py
```

Este comando recria os diagramas EN/PT com o *synthetic blur* antes do downsampling.

---

## Gerar diagrama do modelo de difusão

```bash
python create_diffusion_diagram.py
```

---

## Treinar diffusion U-Net (from scratch)

```bash
python run_diffusion_training.py \
  --train-glob "data/train_94t/*.nii*" \
  --output-dir results/diffusion/training/diffusion_unet \
  --image-size 256 \
  --scale 2 \
  --num-epochs 50
```

---

## Avaliar checkpoint diffusion

```bash
python run_diffusion_inference.py \
  --checkpoint results/diffusion/training/diffusion_unet/best_model.pt \
  --input-path data/sub-0_ses-1_T1w.nii \
  --output-dir results/diffusion/evaluation/diffusion_eval \
  --slice-axis sagittal
```

---

## Resultados gerados

### `metrics_by_slice.csv`

Contém métricas para cada fatia e cada método.

Colunas principais:

```text
slice_axis
slice_index
method
ssim
psnr
mse
mae
rmse
nrmse
pearson
gradient_mse
hfen
diff_percent
isnr
issm
hr_height
hr_width
lr_height
lr_width
degradation_time_ms
upscaling_time_ms
metrics_time_ms
total_method_time_ms
```

---

### `metrics_aggregate.csv`

Contém média e desvio padrão das métricas por método dentro de uma orientação anatómica.

Este ficheiro permite comparar os métodos numa mesma orientação.

---

### `all_axes_aggregate.csv`

Junta os resultados agregados das três orientações anatómicas.

Permite comparar método e orientação ao mesmo tempo.

Localização:

```text
results/analysis/final_comparison/all_axes_aggregate.csv
```

---

### `results/analysis/final_comparison/tables/*`

Conjunto de tabelas para relatório:

- ranking composto por orientação+método;
- ranking global apenas por método (agregando eixos);
- tabela de tamanhos HR/LR;
- tabela de tempos médios de processamento;
- template para comparação de PSNR com literatura.

---

### `final_results_summary.md`

Documento final de resultados em formato Markdown.

Resume:

- melhor método por orientação;
- ranking global;
- principais métricas;
- notas de interpretação.

---

## Experiências planeadas

### Baseline principal

```python
slice_axis = "sagittal"
num_slices = 9
scale = 4
blur_sigma = 1.0
noise_sigma = 0.0
```

---

### Comparação por orientação anatómica

```python
slice_axis = "sagittal"
slice_axis = "coronal"
slice_axis = "axial"
```

---

### Comparação por fator de escala

```python
scale = 2
scale = 4
scale = 8
```

A expectativa é:

```text
x2 → menor degradação
x4 → degradação intermédia
x8 → degradação mais severa
```

---

## Interpretação dos resultados

A análise não deve depender de uma única métrica.

Um método será considerado melhor se apresentar de forma consistente:

- SSIM mais alto;
- PSNR mais alto;
- MSE, MAE e RMSE mais baixos;
- HFEN mais baixo;
- Gradient MSE mais baixo;
- menor erro nos mapas visuais;
- boa preservação visual das estruturas anatómicas.

Se, por exemplo, Lanczos apresentar melhores resultados, a conclusão correta não é que recuperou informação perdida. A conclusão correta é que produziu a reconstrução mais semelhante à referência entre os métodos clássicos avaliados.

---

## Limitações atuais

Esta fase tem várias limitações:

- a imagem LR é criada artificialmente;
- é usado um número limitado de volumes;
- os métodos testados são interpoladores clássicos;
- a estrutura diffusion U-Net já está pronta, mas precisa de treino/validação extensivos com mais dados;
- as métricas são calculadas em fatias 2D;
- não existe validação clínica por especialista;
- a normalização altera a escala original das intensidades MRI;
- bons valores métricos não garantem relevância clínica.

Estas limitações são esperadas, porque esta fase serve como baseline experimental.
