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
- Method: `lanczos`
- SSIM mean: `0.5800`
- PSNR mean: `18.42`
- MSE mean: `0.014402`
- HFEN mean: `0.4643`

## Global ranking

| Rank | Orientation | Method | SSIM mean | PSNR mean | MSE mean | HFEN mean |
|---:|---|---|---:|---:|---:|---:|
| 1 | `axial` | `lanczos` | 0.5800 | 18.42 | 0.014402 | 0.4643 |
| 2 | `axial` | `bicubic` | 0.5758 | 18.34 | 0.014657 | 0.4881 |
| 3 | `coronal` | `lanczos` | 0.5656 | 18.81 | 0.013148 | 0.4635 |
| 4 | `coronal` | `bicubic` | 0.5604 | 18.73 | 0.013403 | 0.4900 |
| 5 | `axial` | `nearest` | 0.5375 | 17.61 | 0.017339 | 0.5189 |
| 6 | `axial` | `bilinear` | 0.5335 | 17.82 | 0.016514 | 0.5880 |
| 7 | `coronal` | `bilinear` | 0.5251 | 18.25 | 0.014965 | 0.5898 |
| 8 | `coronal` | `nearest` | 0.5251 | 18.03 | 0.015735 | 0.5240 |
| 9 | `sagittal` | `lanczos` | 0.4846 | 17.29 | 0.018655 | 0.5132 |
| 10 | `sagittal` | `bicubic` | 0.4791 | 17.23 | 0.018947 | 0.5377 |
| 11 | `sagittal` | `nearest` | 0.4535 | 16.75 | 0.021133 | 0.5615 |
| 12 | `sagittal` | `bilinear` | 0.4385 | 16.83 | 0.020740 | 0.6324 |

## Best method per anatomical orientation

### sagittal

- Best method: `lanczos`
- SSIM mean: `0.4846`
- PSNR mean: `17.29`
- MSE mean: `0.018655`
- HFEN mean: `0.5132`

### coronal

- Best method: `lanczos`
- SSIM mean: `0.5656`
- PSNR mean: `18.81`
- MSE mean: `0.013148`
- HFEN mean: `0.4635`

### axial

- Best method: `lanczos`
- SSIM mean: `0.5800`
- PSNR mean: `18.42`
- MSE mean: `0.014402`
- HFEN mean: `0.4643`

## Interpretation notes

- Higher SSIM indicates better structural similarity between HR and SR.
- Higher PSNR indicates lower global reconstruction error.
- Lower MSE/RMSE/MAE indicates lower pixel-level error.
- Lower HFEN suggests better preservation of high-frequency detail.
- The best classical method should not be interpreted as recovering lost anatomical information. It only produced the closest interpolation-based reconstruction under this controlled degradation setting.
