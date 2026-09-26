# SciCode OOD Split

## 划分方案：Domain-level OOD

### 思路
与 IID split (mydev/mytest) 互补，构造两套独立的 setup：

| Setup | Train | Test | 说明 |
|---|---|---|---|
| **IID** | mydev (20) | mytest (60) | Domain+Difficulty 分层采样，域内划分 |
| **OOD** | ood_train (64) | ood_test (16) | 完整 domain 级别 holdout |

### OOD 划分规则

- **训练集** (`ood_train.jsonl`, 64 题)：Physics + Math + Material Science
- **测试集** (`ood_test.jsonl`, 16 题)：Chemistry + Biology（训练时完全不可见）

```
OOD Train (64):  Physics (37) + Math (14) + Material Science (13)
OOD Test  (16):  Chemistry (8) + Biology (8)
```

### 测试集 16 题明细

| ID | Problem | Domain | Subfield | Difficulty |
|---|---|---|---|---|
| 10 | ewald_summation | Chemistry | Computational Chemistry | Hard |
| 12 | Schrodinger_DFT_with_SCF | Chemistry | Quantum Chemistry | Hard |
| 16 | Davidson_method | Chemistry | Computational Chemistry | Easy |
| 25 | CRM_in_chemostat | Biology | Ecology | Medium |
| 26 | CRM_in_serial_dilution | Biology | Ecology | Medium |
| 30 | helium_slater_jastrow_wavefunction | Chemistry | Quantum Chemistry | Medium |
| 41 | Structural_stability_in_serial_dilution | Biology | Ecology | Medium |
| 44 | two_mer_entropy | Biology | Biochemistry | Medium |
| 46 | helium_atom_vmc | Chemistry | Quantum Chemistry | Medium |
| 53 | Stochastic_Lotka_Volterra | Biology | Ecology | Medium |
| 55 | Swift_Hohenberg | Biology | Ecology | Medium |
| 56 | temporal_niches | Biology | Ecology | Medium |
| 60 | Widom_particle_insertion | Chemistry | Computational Chemistry | Medium |
| 66 | kolmogorov_crespi_potential | Chemistry | Quantum Chemistry | Hard |
| 68 | helium_atom_dmc | Chemistry | Quantum Chemistry | Hard |
| 76 | protein_dna_binding | Biology | Genetics | Medium |

### OOD test 难度分布

| Difficulty | Count |
|---|---|
| Easy | 1 |
| Medium | 11 |
| Hard | 4 |

### 选择 Chemistry + Biology 的理由

1. **语义距离最大**：Chemistry（分子/量子化学）和 Biology（生态/生命系统）与训练集的 Physics/Math/MaterialScience 在问题类型、依赖库、计算范式上差异明显
2. **规模均衡**：两个域各 8 题，合计 16 题（20%），与 SRDD OOD 比例对齐
3. **难度覆盖**：Chemistry Hard 题多（4 Hard），Biology 以 Medium 为主（7 Medium），整体分布合理
4. **可论证性**：Domain-level holdout 边界清晰，reviewer 容易接受

### 局限性声明（论文写作参考）

SciCode 总量仅 80 题，OOD test 仅 16 题，统计显著性有限。建议在论文中：
- 将 SciCode OOD 定位为**补充实验**（supplementary）
- 以 SRDD OOD（240 题）作为**主要 OOD 实验**
- SciCode OOD 用于说明方法在不同 benchmark 上的一致性
