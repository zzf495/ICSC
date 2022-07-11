# Domain Adaptation via Incremental Confidence Samples into Classification

This repository contains the codes of the [ICSC](http://doi.org/10.1002/int.22629) proposed in the International Journal of Intelligent Systems (IJIS). 

Apart from this, I would like to introduce [another repository](https://github.com/zzf495/Re-implementations-of-SDA) that integrates some shallow domain adaptation methods. I hope this can help.

## Code files (matlab implementation)

ICSC.m : the implementation of ICSC;

ACDA.m: the implementation of Adaptive Adjustment;

demo.m: experimental demonstration on Office (surf) dataset.

## Results on datasets

There are some results conducted on different datasets can be directly cited.

### 1. Office+Caltech (Surf)

| SOURCE  | TARGET |       |
| ------- | ------ | ----- |
| C       | A      | 55.53 |
| C       | W      | 59.32 |
| C       | D      | 54.14 |
| A       | C      | 42.21 |
| A       | W      | 54.24 |
| A       | D      | 50.96 |
| W       | C      | 36.69 |
| W       | A      | 40.08 |
| W       | D      | 75.8  |
| D       | C      | 34.19 |
| D       | A      | 40.5  |
| D       | W      | 85.42 |
| AVERAGE |        | 52.42 |

### 2. Office+Caltech (Decaf)

| SOURCE  | TARGET |       |
| ------- | ------ | ----- |
| C       | A      | 93.32 |
| C       | W      | 93.22 |
| C       | D      | 92.36 |
| A       | C      | 88.78 |
| A       | W      | 89.15 |
| A       | D      | 96.18 |
| W       | C      | 83.7  |
| W       | A      | 88.41 |
| W       | D      | 99.36 |
| D       | C      | 82.9  |
| D       | A      | 89.67 |
| D       | W      | 98.98 |
| AVERAGE |        | 91.34 |

### 3. ImageNet +VOC2007

| SOURCE | TARGET  | ICSC  |
| ------ | ------- | ----- |
| I      | V       | 65.05 |
| V      | I       | 81.95 |
|        | AVERAGE | 73.5  |

### 4. OfficeHome (ResNet50)

| SOURCE  | TARGET |       |
| ------- | ------ | ----- |
| Ar      | Cl     | 51.7  |
| Ar      | Pr     | 71.3  |
| Ar      | Re     | 75.7  |
| Cl      | Ar     | 62    |
| Cl      | Pr     | 70.7  |
| Cl      | Re     | 70.7  |
| Pr      | Ar     | 62.4  |
| Pr      | Cl     | 50    |
| Pr      | Re     | 76    |
| Re      | Ar     | 68.2  |
| Re      | Cl     | 52.4  |
| Re      | Pr     | 79    |
| AVERAGE |        | 65.84 |

### 5. Office31 (ResNet50)

| SOURCE  | TARGET |       |
| ------- | ------ | ----- |
| A       | D      | 87.80 |
| A       | W      | 87.80 |
| D       | A      | 74.10 |
| D       | W      | 95.00 |
| W       | A      | 74.30 |
| W       | D      | 97.40 |
| AVERAGE |        | 86.07 |

### 6. ImageCLEF-DA (ResNet50)

| SOURCE  | TARGET |       |
| ------- | ------ | ----- |
| C       | I      | 93.00 |
| C       | P      | 79.86 |
| I       | C      | 95.00 |
| I       | P      | 80.71 |
| P       | C      | 94.17 |
| P       | I      | 93.00 |
| AVERAGE |        | 89.29 |

## Citation

> MLA: Teng, Shaohua, et al. "Domain adaptation via incremental confidence samples into classification." *International Journal of Intelligent Systems* 37.1 (2022): 365-385.

```
@article{teng2022domain,
  title={Domain adaptation via incremental confidence samples into classification},
  author={Teng, Shaohua and Zheng, Zefeng and Wu, Naiqi and Fei, Lunke and Zhang, Wei},
  journal={International Journal of Intelligent Systems},
  volume={37},
  number={1},
  pages={365--385},
  year={2022},
  publisher={Wiley Online Library}
}
```
