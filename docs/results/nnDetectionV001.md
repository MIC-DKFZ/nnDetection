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
| nnUNetPlus  | 0.439<sup>*</sup>    | 0.700    | 0.955    | 0.935    |
| nnUNetBasic | 0.411<sup>*</sup>    | 0.667    | 0.930    | 0.908    |

<sup>*</sup> results with corrected numerical values in softdice loss and improved multi-class import.

&nbsp;

### Validation Pool [AP @ IoU 0.1]

5 Fold Cross Validation

| Model       | ADAM     | ProstateX              | Pancreas | Hepatic Vessel   | Colon | Liver |
|:-----------:|:--------:|:----------------------:|:--------:|:----------------:|:-----:|:-----:|
| nnDetection | 0.780    | 0.300                  | 0.766    | 0.727            | 0.662 | 0.628 |
| nnUNetPlus  | 0.720    | 0.220<sup>*</sup>      | 0.721    | 0.721            | 0.579 | 0.678 |
| nnUNetBasic | 0.657    | 0.202<sup>*</sup>      | 0.691    | 0.699            | 0.509 | 0.567 |

<sup>*</sup>improved multi-class import

&nbsp;

Test Split

| Model       | ProstateX              | Pancreas | Hepatic Vessel   | Colon | Liver |
|:-----------:|:----------------------:|:--------:|:----------------:|:-----:|:-----:|
| nnDetection | 0.221                  | 0.791    | 0.664            | 0.696 | 0.790 |
| nnUNetPlus  | 0.123<sup>*</sup>      | 0.704    | 0.684            | 0.731 | 0.760 |

ADAM Results are listed under Benchmarks
<sup>*</sup>improved multi-class import

&nbsp;

### Test Pool [AP @ IoU 0.1]

5 Fold Cross Validation

| Model       | Abdominal Lymph Nodes | Mediastinal Lymph Nodes |
|:-----------:|:---------------------:|:-----------------------:|
| nnDetection | 0.493                 | 0.440                   |
| nnUNetPlus  | 0.378                 | 0.334                   |
| nnUNetBasic | 0.360                 | 0.302                   |

&nbsp;

Test Split

| Model       | Abdominal Lymph Nodes | Mediastinal Lymph Nodes |
|:-----------:|:---------------------:|:-----------------------:|
| nnDetection | 0.470                 | 0.500                   |
| nnUNetPlus  | 0.311                 | 0.342                   |

Luna results are listed under Benchmarks

&nbsp;

</div>

#### References
- S. G. Armato III, G. McLennan, L. Bidaut, M. F. McNitt-Gray, C. R. Meyer, A. P.Reeves, B. Zhao, D. R. Aberle, C. I. Henschke, E. A. Hoffman, et al.  The lungimage  database  consortium  (lidc)  and  image  database  resource  initiative  (idri):a  completed  reference  database  of  lung  nodules  on  ct  scans.Medical physics,38(2):915–931, 2011
- L. Jin, J. Yang, K. Kuang, B. Ni, Y. Gao, Y. Sun, P. Gao, W. Ma, M. Tan, H. Kang,J.  Chen,  and  M.  Li.   Deep-learning-assisted  detection  and  segmentation  of  ribfractures from CT scans: Development and validation of FracNet.  62.  Publisher:Elsevier
- C.  Tabea  Kossen,  L.  Kaufhold,  M.  H ̈ullebrand,  J.-M.  Kuhnigk,  J.  Br ̈uhning,J. Schaller, B. Pfahringer, A. Spuler, L. Goubergrits, and A. Hennemuth. Cerebralaneurysm detection and analysis, Mar. 2020
- K. Timmins, E. Bennink, I. van der Schaaf, B. Velthuis, Y. Ruigrok, and H. Kuijf.Intracranial Aneurysm Detection and Segmentation Challenge, Mar. 2020.
- N.  Heller,  N.  Sathianathen,  A.  Kalapara,  E.  Walczak,  K.  Moore,  H.  Kaluzniak,J. Rosenberg, P. Blake, Z. Rengel, M. Oestreich, et al.  The kits19 challenge data:300 kidney tumor cases with clinical context, ct semantic segmentations, and sur-gical outcomes.arXiv preprint arXiv:1904.00445, 2019
- G. Litjens, O. Debats, J. Barentsz, N. Karssemeijer, and H. Huisman.  Computer-aided detection of prostate cancer in mri.IEEE TMI, 33(5):1083–1092, 2014
- R.  Cuocolo,  A.  Comelli,  A.  Stefano,  V.  Benfante,  N.  Dahiya,  A.  Stanzione,A.  Castaldo,  D.  R.  D.  Lucia,  A.  Yezzi,  and  M.  Imbriaco.   Deep  learning  whole-gland and zonal prostate segmentation on a public mri dataset.Journal of Mag-netic Resonance Imaging, 2021.
- A. L. Simpson, M. Antonelli, S. Bakas, M. Bilello, K. Farahani, B. Van Ginneken,A. Kopp-Schneider, B. A. Landman, G. Litjens, B. Menze, et al.  A large anno-tated medical image dataset for the development and evaluation of segmentationalgorithms.arXiv preprint arXiv:1902.09063, 2019.
- H. R. Roth, L. Lu, A. Seff, K. M. Cherry, J. Hoffman, S. Wang, J. Liu, E. Turkbey,and R. M. Summers.  A new 2.5 d representation for lymph node detection usingrandom sets of deep convolutional neural network observations. InMICCAI, pages520–527. Springer, 2014
- A. Seff, L. Lu, A. Barbu, H. Roth, H.-C. Shin, and R. M. Summers. Leveraging mid-level semantic boundary cues for automated lymph node detection.  InMICCAI,pages 53–61. Springer, 2015

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
- A. A. A. Setio, A. Traverso, T. de Bel, M. S. Berens, C. van den Bogaard, P. Cerello,H.  Chen,  Q.  Dou,  M.  E.  Fantacci,  B.  Geurts,  R.  van  der  Gugten,  P.  A.  Heng,B. Jansen, M. M. de Kaste, V. Kotov, J. Y.-H. Lin, J. T. Manders, A. S ́o ̃nora-Mengana, J. C. Garc ́ıa-Naranjo, E. Papavasileiou, M. Prokop, M. Saletta, C. M.Schaefer-Prokop, E. T. Scholten, L. Scholten, M. M. Snoeren, E. L. Torres, J. Van-demeulebroucke,  N.  Walasek,  G.  C.  Zuidhof,  B.  van  Ginneken,  and  C.  Jacobs.Validation, comparison, and combination of algorithms for automatic detection ofpulmonary nodules in computed tomography images: The luna16 challenge.Me-dIA, 42:1–13, 2017.
- Z. Gong, D. Li, J. Lin, Y. Zhang and K. -M. Lam, "Towards Accurate Pulmonary Nodule Detection by Representing Nodules as Points With High-Resolution Network," in IEEE Access, vol. 8, pp. 157391-157402, 2020, doi: 10.1109/ACCESS.2020.3019104
- Q. Dou, H. Chen, L. Yu, J. Qin and P. Heng, "Multilevel Contextual 3-D CNNs for False Positive Reduction in Pulmonary Nodule Detection," in IEEE Transactions on Biomedical Engineering, vol. 64, no. 7, pp. 1558-1567, July 2017, doi: 10.1109/TBME.2016.2613502.
- Gupta, A., Saar, T., Martens, O. and Moullec, Y.L. (2018), Automatic detection of multisize pulmonary nodules in CT images: Large-scale validation of the false-positive reduction step. Med. Phys., 45: 1135-1149. https://doi.org/10.1002/mp.12746
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
