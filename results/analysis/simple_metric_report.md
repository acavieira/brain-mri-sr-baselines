# Metric Report

## Overall Winners

| Metric | Direction | Best axis | Best method | Value |
|---|---|---|---|---:|
| `ssim_mean` | higher | `axial` | `bicubic` | 0.683806 |
| `psnr_mean` | higher | `axial` | `lanczos` | 20.812283 |
| `mse_mean` | lower | `axial` | `lanczos` | 0.008328 |
| `mae_mean` | lower | `axial` | `lanczos` | 0.056282 |
| `rmse_mean` | lower | `axial` | `lanczos` | 0.091164 |
| `nrmse_mean` | lower | `axial` | `lanczos` | 0.091164 |
| `pearson_mean` | higher | `axial` | `lanczos` | 0.955173 |
| `gradient_mse_mean` | lower | `coronal` | `lanczos` | 0.137836 |
| `hfen_mean` | lower | `axial` | `lanczos` | 0.151810 |
| `diff_percent_mean` | lower | `axial` | `lanczos` | 5.628154 |

## Best Method Per Axis

| Axis | Best method | SSIM mean | PSNR mean | MSE mean | HFEN mean |
|---|---|---:|---:|---:|---:|
| `sagittal` | `bicubic` | 0.6567 | 20.14 | 0.009711 | 0.1626 |
| `coronal` | `bicubic` | 0.6702 | 20.74 | 0.008439 | 0.1606 |
| `axial` | `bicubic` | 0.6838 | 20.77 | 0.008410 | 0.1532 |