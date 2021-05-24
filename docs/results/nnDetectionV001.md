# nnDetection v0.1
## Pool v0.1

<div align="center">

<img src=#TODO width="600px">

</div>

### Train Pool [AP @ IoU 0.1]

5 Fold Cross Validation

| Model       | LIDC     | RibFrac  | CADA     | Kits19   |
|:-----------:|:--------:|:--------:|:--------:|:--------:|
| nnDetection | 0.605    | 0.765    | 0.924    | 0.923    |
| nnUNetPlus  | 0.304    | 0.700    | 0.955    | 0.935    |
| nnUNetBasic | 0.232    | 0.667    | 0.930    | 0.908    |

### Validation Pool [AP @ IoU 0.1]

5 Fold Cross Validation

| Model       | ADAM     | ProstateX  | Pancreas | Hepatic Vessel   | Colon | Liver |
|:-----------:|:--------:|:----------:|:--------:|:----------------:|:-----:|:-----:|
| nnDetection | 0.780    | 0.300      | 0.766    | 0.727            | 0.662 | 0.628 |
| nnUNetPlus  | 0.720    | 0.197      | 0.721    | 0.721            | 0.579 | 0.678 |
| nnUNetBasic | 0.657    | 0.204      | 0.691    | 0.699            | 0.509 | 0.567 |

Test Split

| Model       | ProstateX  | Pancreas | Hepatic Vessel   | Colon | Liver |
|:-----------:|:----------:|:--------:|:----------------:|:-----:|:-----:|
| nnDetection | 0.221      | 0.791    | 0.664            | 0.696 | 0.790 |
| nnUNetPlus  | 0.078      | 0.704    | 0.684            | 0.731 | 0.760 |

ADAM Results are listed under Benchmarks

### Test Pool [AP @ IoU 0.1]

5 Fold Cross Validation

| Model       | Lymph Nodes |
|:-----------:|:-----------:|
| nnDetection | 0.205       |
| nnUNetPlus  | 0.162       |
| nnUNetBasic | 0.159       |

Test Split

| Model       | Lymph Nodes |
|:-----------:|:-----------:|
| nnDetection | 0.270       |
| nnUNetPlus  | 0.169       |

Luna results are listed under Benchmarks

## Benchmarks
### Luna

<div align="center">

<img src=#TODO width="600px">

</div>

| Methods     | 1/8   | 1/4   | 1/2   | 1     | 2     | 4     | 8     | CPM   |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| nnDetection | 0.812 | 0.885 | 0.927 | 0.950 | 0.969 | 0.979 | 0.985 | 0.930 |

### ADAM Live Leaderboard

<div align="center">

<img src=#TODO width="600px">

</div>

| Model       | Sens | FP  |
|:-----------:|:----:|:---:|
| nnDetection | 0.64 | 0.3 |
