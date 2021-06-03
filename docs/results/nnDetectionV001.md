# nnDetection v0.1
## Pool v0.1

<div align="center">

<img src=./source/v001/results_full_final.svg width="800px">


&nbsp;

### Train Pool [AP @ IoU 0.1]

5 Fold Cross Validation

| Model       | LIDC                 | RibFrac  | CADA     | Kits19   |
|:-----------:|:--------------------:|:--------:|:--------:|:--------:|
| nnDetection | 0.605                | 0.765    | 0.924    | 0.923    |
| nnUNetPlus  | 0.385<sup>*</sup>    | 0.700    | 0.955    | 0.935    |
| nnUNetBasic | 0.346<sup>*</sup>    | 0.667    | 0.930    | 0.908    |

<sup>*</sup> results with corrected numerical values in softdice loss. Out of the box results: nnUNetPlus 0.304 and nnUNetBasic 0.232

&nbsp;

### Validation Pool [AP @ IoU 0.1]

5 Fold Cross Validation

| Model       | ADAM     | ProstateX  | Pancreas | Hepatic Vessel   | Colon | Liver |
|:-----------:|:--------:|:----------:|:--------:|:----------------:|:-----:|:-----:|
| nnDetection | 0.780    | 0.300      | 0.766    | 0.727            | 0.662 | 0.628 |
| nnUNetPlus  | 0.720    | 0.197      | 0.721    | 0.721            | 0.579 | 0.678 |
| nnUNetBasic | 0.657    | 0.204      | 0.691    | 0.699            | 0.509 | 0.567 |

&nbsp;

Test Split

| Model       | ProstateX  | Pancreas | Hepatic Vessel   | Colon | Liver |
|:-----------:|:----------:|:--------:|:----------------:|:-----:|:-----:|
| nnDetection | 0.221      | 0.791    | 0.664            | 0.696 | 0.790 |
| nnUNetPlus  | 0.078      | 0.704    | 0.684            | 0.731 | 0.760 |

ADAM Results are listed under Benchmarks

&nbsp;

### Test Pool [AP @ IoU 0.1]

5 Fold Cross Validation

| Model       | Lymph Nodes |
|:-----------:|:-----------:|
| nnDetection | 0.205       |
| nnUNetPlus  | 0.162       |
| nnUNetBasic | 0.159       |

&nbsp;

Test Split

| Model       | Lymph Nodes |
|:-----------:|:-----------:|
| nnDetection | 0.270       |
| nnUNetPlus  | 0.169       |

Luna results are listed under Benchmarks

&nbsp;

</div>

## Benchmarks
### Luna
Disclaimer:
This overview reflects the literature upon submission of nnDetection (March 2021).
It will not be updated with newer methods and can not replace a thorough literature research of future work.

<div align="center">

<img src=./source/v001/luna.png width="800px">

| Methods                      | 1/8   | 1/4   | 1/2   | 1     | 2     | 4     | 8     | CPM   |
|:----------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
Dou et al.   (2017a)           | 0.692 | 0.745 | 0.819 | 0.865 | 0.906 | 0.933 | 0.946 | 0.839 |
Zhu et al.   (2018)            | 0.692 | 0.769 | 0.824 | 0.865 | 0.893 | 0.917 | 0.933 | 0.842 |
Wang et al.  (2018)            | 0.676 | 0.776 | 0.879 | 0.949 | 0.958 | 0.958 | 0.958 | 0.878 |
Ding et al.  (2017)            | 0.748 | 0.853 | 0.887 | 0.922 | 0.938 | 0.944 | 0.946 | 0.891 |
Khosravan et al. (2018)        | 0.709 | 0.836 | 0.921 | 0.953 | 0.953 | 0.953 | 0.953 | 0.897 |
Cao et al. (2020)              | 0.868 | 0.900 | 0.913 | 0.915 | 0.916 | 0.931 | 0.932 | 0.911 |
Liu et al. (2019)              | 0.848 | 0.876 | 0.905 | 0.933 | 0.943 | 0.957 | 0.970 | 0.919 |
Song et al. (2020)             | 0.723 | 0.838 | 0.887 | 0.911 | 0.928 | 0.934 | 0.948 | 0.881 |
nnDetection v0.1 (ours, 2021)  | 0.812 | 0.885 | 0.927 | 0.950 | 0.969 | 0.979 | 0.985 | 0.930 |
*Methods with FPR*<sup>*</sup> |       |       |       |       |       |       |       |       |
Cao et al. (2020) + FPR        | 0.848 | 0.899 | 0.925 | 0.936 | 0.949 | 0.957 | 0.960 | 0.925 |
Liu et al. (2019) + FPR        | 0.904 | 0.914 | 0.933 | 0.957 | 0.971 | 0.971 | 0.971 | 0.952 |

<sup>*</sup> Some of the other methods also use FPR stages but the methods listed below report results w. and wo. FPR.

&nbsp;

</div>

#### References (no particular oder)
- J. Ding, A. Li, Z. Hu, and L. Wang. Accurate pulmonary nodule detection in computed tomography images using deep convolutional neural networks. In MICCAI, pages 559–567. Springer, 2017
- Q. Dou, H. Chen, Y. Jin, H. Lin, J. Qin, and P.-A. Heng. Automated pulmonary nodule detection via 3d convnets with online sample filtering and hybrid-loss residual learning. In MICCAI, pages 630–638. Springer, 2017
- N. Khosravan and U. Bagci. S4nd: Single-shot single-scale lung nodule detection. In MICCAI, pages 794–802. Springer, 2018.
- B. Wang, G. Qi, S. Tang, L. Zhang, L. Deng, and Y. Zhang. Automated pulmonary nodule detection: High sensitivity with few candidates. In MICCAI, pages 759–767. Springer, 2018
- W. Zhu, C. Liu, W. Fan, and X. Xie. Deeplung: Deep 3d dual path nets for automated pulmonary nodule detection and classification. In WACV, pages 673–681. IEEE, 2018
- J. Liu, L. Cao, O. Akin, and Y. Tian. 3dfpn-hs: 3d feature pyramid network based high sensitivity and specificity pulmonary nodule detection. In MICCAI, pages 513–521. Springer, 2019
-  T. Song, J. Chen, X. Luo, Y. Huang, X. Liu, N. Huang, Y. Chen, Z. Ye, H. Sheng, S. Zhang, and G. Wang. CPM-net: A 3d center-points matching network for pulmonary nodule detection in CT scans. In A. L. Martel, P. Abolmaesumi, D. Stoyanov, D. Mateus, M. A. Zuluaga, S. K. Zhou, D. Racoceanu, and L. Joskowicz, editors, MICCAI, pages 550–559. Springer International Publishing
- H. Cao, H. Liu, E. Song, G. Ma, X. Xu, R. Jin, T. Liu, and C. C. Hung. A twostage convolutional neural networks for lung nodule detection. IEEE Journal of Biomedical and Health Informatics, 24(7):2006–2015, 2020.

### ADAM Live Leaderboard
Disclaimer:
This overview reflects the literature upon submission of nnDetection (March 2021).
It will not be updated with newer methods and can not replace a thorough literature research of future work.

<div align="center">

<img src=./source/v001/adam.svg width="300">


| Model       | Sens | FP  |
|:-----------:|:----:|:---:|
| nnDetection | 0.64 | 0.3 |

</div>
