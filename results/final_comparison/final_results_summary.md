# Final Results Summary

This document summarizes the aggregate results obtained across the three anatomical orientations.

Ranking criterion used in this summary:

```text
1. Higher mean SSIM
2. Higher mean PSNR
3. Lower mean MSE
4. Lower mean HFEN
```

## Best overall result

- Orientation: `axial`
- Method: `bicubic`
- SSIM mean: `0.6838`
- PSNR mean: `20.77`
- MSE mean: `0.008410`
- HFEN mean: `0.1532`

## Global ranking

| Rank | Orientation | Method | SSIM mean | PSNR mean | MSE mean | HFEN mean |
|---:|---|---|---:|---:|---:|---:|
| 1 | `axial` | `bicubic` | 0.6838 | 20.77 | 0.008410 | 0.1532 |
| 2 | `axial` | `lanczos` | 0.6817 | 20.81 | 0.008328 | 0.1518 |
| 3 | `coronal` | `bicubic` | 0.6702 | 20.74 | 0.008439 | 0.1606 |
| 4 | `coronal` | `lanczos` | 0.6685 | 20.76 | 0.008393 | 0.1595 |
| 5 | `axial` | `nearest` | 0.6635 | 20.19 | 0.009600 | 0.1758 |
| 6 | `sagittal` | `bicubic` | 0.6567 | 20.14 | 0.009711 | 0.1626 |
| 7 | `sagittal` | `lanczos` | 0.6565 | 20.19 | 0.009602 | 0.1602 |
| 8 | `axial` | `bilinear` | 0.6556 | 20.17 | 0.009652 | 0.2410 |
| 9 | `coronal` | `nearest` | 0.6542 | 20.26 | 0.009435 | 0.1823 |
| 10 | `coronal` | `bilinear` | 0.6442 | 20.21 | 0.009539 | 0.2459 |
| 11 | `sagittal` | `nearest` | 0.6406 | 19.58 | 0.011050 | 0.1877 |
| 12 | `sagittal` | `bilinear` | 0.6164 | 19.52 | 0.011208 | 0.2599 |

## Best method per anatomical orientation

### sagittal

- Best method: `bicubic`
- SSIM mean: `0.6567`
- PSNR mean: `20.14`
- MSE mean: `0.009711`
- HFEN mean: `0.1626`

### coronal

- Best method: `bicubic`
- SSIM mean: `0.6702`
- PSNR mean: `20.74`
- MSE mean: `0.008439`
- HFEN mean: `0.1606`

### axial

- Best method: `bicubic`
- SSIM mean: `0.6838`
- PSNR mean: `20.77`
- MSE mean: `0.008410`
- HFEN mean: `0.1532`

## Interpretation notes

- Higher SSIM indicates better structural similarity between HR and SR.
- Higher PSNR indicates lower global reconstruction error.
- Lower MSE/RMSE/MAE indicates lower pixel-level error.
- Lower HFEN suggests better preservation of high-frequency detail.
- The best classical method should not be interpreted as recovering lost anatomical information. It only produced the closest interpolation-based reconstruction under this controlled degradation setting.
