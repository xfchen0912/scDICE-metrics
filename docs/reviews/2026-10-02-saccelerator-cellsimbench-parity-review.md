# 原版一致性审查：SACCELERATOR / CellSimBench 两个新 benchmark 模块

- 日期：2026-10-02
- 审查对象：`scDICE-metrics` HEAD `02fb1b2`（"Add SACCELERATOR-style spatial metrics and consensus helpers."）
  - `4bfdea4` "Add CellSimBench-style counterfactual metrics and DRF calibration."（其前一提交）
- 上游原版
  - SACCELERATOR：`SpatialHackathon/SACCELERATOR` @ `dac468d`（本地 `tmp/SACCELERATOR`）
  - CellSimBench：`shiftbioscience/Perturbation-Models-Outperform-Baselines` @ `15361ae`（本地 `tmp/Perturbation-Models-Outperform-Baselines`，原版实现在 `cellsimbench/core/`）
- 验证环境：Python 3.10 + numpy 1.23.5 / scipy 1.12.0 / sklearn 1.4.0 / pandas 2.2.2；R 4.5.0（`clue`、`jsonlite`、`cluster` 可用）
- **修订记录（2026-10-02）**：经独立复核（见同目录 `2026-10-02-saccelerator-cellsimbench-parity-review-verification.md`），修正了 2.5 节关于 `smoothness_entropy` 邻域的结论（原版 `dbscan::kNN` 同样排除自身，差异只有对数底数），并收紧 3.3 节 `pathway_recovery` 的归类与 2.2 节的机制表述。P0/P1 结论未变。
- **后续（2026-10-02）**：`pas` / `chaos` 已改为与 SDMBench `compute_PAS` / `compute_CHAOS` 一致（精确 kNN、原始坐标、`StandardScaler`、跳过 ≤2 细胞簇），不再对齐 SACCELERATOR R。下文 2.4 节描述的是修改前的偏差。
- **后续（2026-10-02）**：`domain_specific_f1` 已按 SACCELERATOR NA 语义补齐未匹配 GT domain（`f1=NaN`，`macro` 随之 NaN）。下文 2.3 节描述的是修改前的偏差。
- **后续（2026-10-02）**：P2 `smoothness_entropy` 已改为自然对数（nats）；邻域仍排除自身、默认 k=6。下文 2.5 节描述的是修改前的 `log2` 偏差。

---

## 0. 结论摘要

| 模块 | 忠实度 | 主要问题 |
| --- | --- | --- |
| CellSimBench-style（`metrics/_cellsimbench.py`、`_calibration.py`、`Counterfactual.cellsimbench()`） | 指标层面基本忠实，覆盖面与 DRF 有缺口 | `drf_ctrl` 在 5 个 `*deltactrl*` 指标上整列丢失；工厂未开 `nir`/`pds`/`*_degs`/`mae*`；`mse`/`mae` 未注册；无 GSEA |
| SACCELERATOR-style（`metrics/_spatial_supervised.py`、`_spatial_unsupervised.py`、`benchmark/_consensus.py`） | **不忠实**，存在计算错误 | `matched_mcc`/`matched_jaccard` 标签编码错位（实测可出现符号翻转）；`domain_specific_f1` 的 NA 语义不同；`pas`/`chaos` 有参数级偏差；真·SpatialARI 缺失；`smoothness_entropy` 口径不同 |

一句话：**CellSimBench 的原子指标可以直接替代原版；SACCELERATOR 的 matched 系列指标当前结果不可用，其余指标存在小幅但系统性的数值偏差。**

---

## 1. 验证方法

两项原版实现都可直接执行，因此本审查不是纯代码比对，而是**逐函数数值对拍**：

1. SACCELERATOR：以原版 `metric/*/` 脚本为基准（`Entropy.py`、`MCC.py`、`jaccard.py` 为 Python；`domain-specific-f1.r`、`PAS.r`、`CHAOS.r` 由 `Rscript` 执行），合成数据写入 TSV 后运行，与 `scdice_metrics` 的函数输出比较。
2. CellSimBench：注入 stub 后直接加载原版 `cellsimbench.core.data_manager` / `cellsimbench.core.metrics_engine`，调用 `DataManager.get_deg_weights/get_deg_mask`、`MetricsEngine._calculate_nir_scores/_calculate_knn_jaccard_scores`、`mse/wmse/mae/wmae/r2_score_on_deltas`，与 port 的同名函数逐个比较。
3. DRF：以原版 `analyses/calibration/calibration_analysis.py:409-560` 的 `calculate_drf` 与基线选择规则为准，与 `metrics/_calibration.py` 逐列比较。

复现脚本见文末附录 A / B。

---

## 2. SACCELERATOR-style 模块

### 2.1 逐指标对照

| 指标（scDICE 名） | 原版位置 | 状态 | 备注 |
| --- | --- | --- | --- |
| `gt_mixture_entropy` | `metric/Entropy/Entropy.py:62` | ✅ 完全一致 | 实测 0.5034869983305348 两侧相同 |
| `spatial_ari` / `spatial_nmi` | `metric/ARI/ARI.py`、`metric/NMI/` | ✅ 一致 | 纯标签版，无匹配；注意名字易与真·SpatialARI 混淆 |
| `domain_specific_f1` | `metric/domain-specific-f1/domain-specific-f1.r` | ⚠️ 部分一致 | `n_clust == n_gt`、`n_clust > n_gt`（含 dummy 补列）时与 R 完全一致；`n_gt > n_clust` 时行为不同（见 2.3） |
| `matched_mcc` | `metric/MCC/MCC.py:60-65` | ❌ **计算错误** | 实测可出现符号翻转（见 2.2） |
| `matched_jaccard` | `metric/jaccard/jaccard.py:60-65` | ❌ **计算错误** | 实测 0.714286 → 0.097222（见 2.2） |
| `calinski_harabasz` / `davies_bouldin` | `metric/Calinski-Harabasz/`、`metric/Davies-Bouldin/` | ✅ 一致 | 同为 sklearn 直调 |
| `cluster_specific_silhouette` | `metric/cluster-specific-silhouette/` | ✅ 一致（取 mean 列） | 原版同时输出 mean/median，port 只出 mean + `macro` |
| `pas` | `metric/PAS/PAS.r:88-110` | ⚠️ 默认参数偏差 | port 默认 `standardize=True`，原版用原始坐标（见 2.4） |
| `chaos` | `metric/CHAOS/CHAOS.r:88-104` | ⚠️ 小偏差 | z-score 的 ddof 不同 + 小簇跳过规则不同（见 2.4） |
| `spatial_external_ari` | `metric/SpatialARI/SpatialARI.r` | ❌ 缺失 | 原版调用 `getSpatialExternalMetrics(..., metrics="SpatialARI")`（空间加权外部 ARI）；port 为占位符恒返回 NaN |
| `smoothness_entropy`（共识层） | `consensus/02_Smoothness_entropy/Smoothness_entropy.r:59-72` | ⚠️ 口径不同 | 邻域一致（都排除自身、k=6），但对数底数不同（见 2.5） |
| `cross_method_ari`（共识层） | `consensus/02_Cross_method_ARI/Cross_method_ARI.r:39-52` | ✅ 等价 | sklearn `adjusted_rand_score` 与 `mclust::adjustedRandIndex` 公式等价 |
| FMI / V_measure / LISI | `metric/FMI`、`metric/V_measure`、`metric/LISI` | ❌ 未实现 | 原版 metric 目录中存在 |

### 2.2 P0：`matched_mcc` / `matched_jaccard` 标签编码错位

位置：`src/scdice_metrics/metrics/_spatial_supervised.py:139`（`matched_mcc`）、`:148`（`matched_jaccard`）。

`match_predicted_to_ground_truth` 返回的是 **GT 标签值**（例如 `"D0"`/`"D1"`）替换后的数组，但随后：

```python
gt_codes, _ = pd.factorize(labels_arr)    # 按 labels 的“值首次出现顺序”编码
pred_codes, _ = pd.factorize(matched)     # 按 matched 的“值首次出现顺序”编码
```

`pd.factorize` 的编码顺序取决于值在数组中的首次出现位置。匹配后的数组与 GT 数组的首次出现顺序通常不同，于是两边整数编码空间错位：GT 的 A 类被拿去和预测的 B 类比较。

机制上更准确的说法是：原版**先在整数编码空间（`cat.codes`）里做匈牙利匹配**，`domains.map(dict(zip(row_ind, col_ind)))` 直接把预测 code 改成 GT code，于是匹配后的预测值与 `y_true` 共用同一套 code；port 则在匹配成字符串标签之后又各自 `factorize` 一次，把已经对齐的信息重新打乱。原版之所以不翻车，关键在"先编码、再在整数空间匹配"，而不在类别名排序本身（虽然 pandas 2.x 对字符串类别确实按名字排序）。

对抗构造（`gt = [D0]×40 + [D1]×100 + [D0]×100`，`pred = [X]×140 + [Y]×100`，匈牙利匹配后 `X→D1`、`Y→D0`）：

| 指标 | 原版 | scDICE port |
| --- | --- | --- |
| MCC | **+0.714286** | **-0.714286**（符号翻转） |
| Jaccard | **0.714286** | **0.097222** |
| Entropy | 0.503487 | 0.503487（正常） |

随机场景（k 个 domain、25% 误分、`n_clust == n_gt`，原版脚本均能正常出值）12 例中有 **4 例严重不符**：

```
trial 4  k=7: MCC 0.7188 → -0.1146   | Jaccard 0.6131 → 0.0240
trial 5  k=5: MCC 0.7438 →  0.1921   | Jaccard 0.6611 → 0.2946
trial 6  k=4: MCC 0.7462 →  0.2315   | Jaccard 0.6829 → 0.3383
trial 8  k=5: MCC 0.8063 →  0.2435   | Jaccard 0.7331 → 0.3329
```

现有测试无法发现该问题：`tests/metrics/test_spatial_saccelerator.py:36` 只在 `labels == pred` 的完美匹配下断言（此时编码偶然对齐）。

补充：原版在 `n_clust > n_gt` 时，`domains.map(dict(zip(row_ind, col_ind)))` 会给未匹配簇写 `NaN`，进而 `matthews_corrcoef` 直接抛 `ValueError: Input y_pred contains NaN`。SACCELERATOR 的 workflow 未传 `--matched_labels`（见 `workflows/04_metrics.smk`），说明其成绩是在"簇数 ≤ domain 数"（路径里的 `{nclust}` 通常取 GT domain 数）下得到的。port 选择保留未匹配簇原始标签、不报错，这是健壮性改进，但**不能据此声称与原版数值一致**，需要在文档中显式声明。

### 2.3 P1：`domain_specific_f1` 的 NA 语义

位置：`src/scdice_metrics/metrics/_spatial_supervised.py:94`。

原版 R 在 `n_gt > n_clust` 时会为"没有任何簇匹配到"的 domain 追加一行 `f1 = NA`（`domain-specific-f1.r:108-120`）；port 只统计被匹配到的 domain，`macro` 也只在被匹配到的 domain 上取均值。

实测（GT 5 类、预测 3 簇）：

```
R 输出 : [["domainB",0.6708],["domainC",0.6735],["domainE",1],[null,"NA"],[null,"NA"]]
port  : {'domainB':0.6708,'domainC':0.6735,'domainE':1.0,'macro':0.7814}
```

被匹配上的逐 domain 数值一致；差异在于 port 无法表达"某个 domain 完全未被预测"，若无额外约定，会显得比原版乐观。

### 2.4 P1：PAS / CHAOS 的参数级偏差

**PAS**（`src/scdice_metrics/metrics/_spatial_clustering.py:80`）

- 原版 `PAS.r:100-110`：原始坐标 + 精确 10-NN（排除自身），`sum(neighbor_label != self_label) > k/2` 判为异常。
- port 坐标回退路径默认 `standardize=True`（z-score 后再算距离）。两轴尺度不同时实测：

```
参考（原版，原始坐标）   = 0.9875
port standardize=True    = 0.9800   ← 默认值偏差
port standardize=False   = 0.9875   ← 与原版一致
```

- 更关键的是 `Benchmarker` 路径：`_spatial_prepare.build_spatial_neighbors` 使用 `pynndescent` **近似** kNN（`nearest_neighbors/_pynndescent.py:27`，`k = max(spatial_n_neighbors, pas_k+1)`，默认 ≥11），而原版是精确 10-NN。

**CHAOS**（`src/scdice_metrics/metrics/_spatial_clustering.py:129-151`）

- 原版 `CHAOS.r:95` 用 `scale()`（样本标准差，ddof=1），port 用 `StandardScaler`（ddof=0）。无小簇时所有距离按同一常数缩放，CHAOS 的比值**精确等于** `sqrt(n/(n-1))`（实测 n=300：`0.2391698650 / 0.2387709159 = 1.0016708449 = sqrt(300/299)`）。
- 原版 `CHAOS.r:101` 的 `if(length(location_cluster)==2) next` 只跳过**单细胞簇**：m=1 时子集退化为长度 2 的向量（`length` 为 2），m≥2 时是 `length = 2m` 的矩阵；port 跳过 `<=2` 个细胞的簇，因此 2 细胞簇被错误地忽略。

```
参考（scale, ddof=1）            = 0.250198
port standardize=True            = 0.250511   （≈ +0.13%）
port standardize=False           = 13.04554   （必须保持 True）
构造 2 细胞簇后：参考 0.254002 vs port 0.249927（≈ -1.6%）
```

结论：PAS/CHAOS 的**排序**基本可用，**数值**不保证与原版一致。`openspec/changes/add-saccelerator-metrics/tasks.md` 中勾选的 "PAS/CHAOS numerical alignment tests vs SACCELERATOR reference" 在仓库中没有任何对应测试或 golden 数据，该勾选需要修正。

### 2.5 P2：共识层 `smoothness_entropy`

位置：`src/scdice_metrics/benchmark/_consensus.py:41`。与原版 `Smoothness_entropy.r:59-72` 有两处实质差异：

1. 邻域：**两侧一致，都排除自身**。原版 `dbscan::kNN(coords, k=6)` 在 `query = NULL` 时返回 6 个邻居且不含 query 点（已用 dbscan 1.1-12 实测：`kNN` 的 `$id` 首列不含自身、`$dist` 无 0 值；CRAN 文档亦称自匹配被移除）；port 的 `np.argsort(dist)[:, 1:k+1]` 同样取 6 个非自身邻居。
2. 对数底数：原版 `calc_entropy` 用自然对数 `log`，port 用 `log2`。

因此数值差异**只来自对数底数**，比例恰为 `1/ln2 ≈ 1.442695`。实测（n=300，k=6）：

```
port (log2, 排除自身) = 1.8034436881
参考 (ln,   排除自身) = 1.2500519077
参考 (log2, 排除自身) = 1.8034436881   ← 与 port 完全相同
参考 (ln,   包含自身) = 1.2426977124   ← 不构成解释
port / 参考(ln) = 1.4426950409 = 1/ln2
```

注：受 `np.argsort` 平局顺序影响，坐标完全重复时 port 的"剔除第 0 列"可能剔错对象；原版是显式剔除自身。属极端边角，单独说明即可。

---

## 3. CellSimBench-style 模块

### 3.1 逐项对照

| 项目 | 原版位置 | 状态 |
| --- | --- | --- |
| `deg_weights_from_scores` | `data_manager.py:183-226`（`_precompute_deg_weights`） | ✅ 逐元素一致（min-max→平方→按基因取 max→reindex） |
| `deg_mask_from_pvals` | `data_manager.py:273-321`（`get_deg_mask(topn=…)`） | ✅ 逐元素一致（按 p 值排序取 top-N→按基因 any→reindex） |
| `profile_mse/profile_mae/profile_wmse/profile_wmae` | `data_manager.py:17-76` | ✅ 一致；全零权重两侧均返回 NaN |
| `r2_delta`（含 `sample_weight`、clip 到 -1） | `data_manager.py:89-110` | ✅ 一致；⚠️ 权重全为 0 时原版退回无加权 R²（-1.0），port 返回 NaN |
| `pearson_deltactrl` / `pearson_deltapert` / `*_degs` | `metrics_engine.py:129-150` | ✅ 代数一致，`dataset_mean`/`reference` 语义与原版一致 |
| `r2_deltactrl` / `r2_deltapert` / `weighted_r2_*` | `metrics_engine.py` | ✅ 一致（含 `deg_mask.sum() <= 2 → NaN` 的守卫） |
| `nir` / `pds` | `metrics_engine.py:369-460` | ✅ 与 `_calculate_nir_scores` 同值（含欧氏/cityblock、组内比较、`<2` 退化） |
| `knn_jaccard_deltapert` | `metrics_engine.py:462-537` | ✅ 与 `_calculate_knn_jaccard_scores` 同值（k=20、排除自身、Jaccard） |
| DRF | `analyses/calibration/calibration_analysis.py:409-560` | ⚠️ `drf_mean`/`drf_interpolated` 正确，**`drf_ctrl` 错**（见 3.2） |
| `pathway_recovery_deltapert`（GSEA） | `metrics_engine.py:36-70` | ❌ 未实现 |
| `deg_recovery_deltapert` | `metrics_engine.py:334-360` | ❌ 未实现（原版 `calculate_all_metrics` 中已注释掉，优先级低） |

实测示例（`/tmp` 对拍脚本，详见附录 B）：

```
deg weights identical: True | deg mask identical: True
MSE  orig/port: 2.4489589412855555 2.4489589412855555
MAE  orig/port: 1.2662229047693494 1.2662229047693494
WMSE orig/port: 2.226817105279431 2.226817105279431
WMAE orig/port: 1.2164537569286742 1.2164537569286742
NIR identical: True          KNN-Jaccard identical: True
weighted r2 orig/port: -0.9342495371626964 -0.9342495371626964
```

### 3.2 P0：DRF 的 `drf_ctrl` 基线选择错误

原版对 `*deltactrl*` 组有专门分支：control 基线会让 deltactrl 的相关系数/R²退化为 NaN，因此 `drf_ctrl` 改用 **mean 基线**（`calibration_analysis.py:428-434` 的 `MEAN_BASELINE_METRICS_FOR_CTRL = [pearson_deltactrl, pearson_deltactrl_degs, r2_deltactrl, r2_deltactrl_degs, weighted_r2_deltactrl]`）。

port 的 `_null_baseline_method_for_metric`（`src/scdice_metrics/metrics/_calibration.py:56`）没有该特例，`drf_all_versions()`（`:125`）对所有指标一律用 `control` 作 `drf_ctrl` 基线。实测：

```
指标            DRF 版本          原版       port
r2_deltapert    drf_mean         +0.6500    +0.6500   ✅
r2_deltapert    drf_interpolated +0.4375    +0.4375   ✅
r2_deltapert    drf_ctrl         +0.4375    +0.4375   ✅
r2_deltactrl    drf_mean         +0.7143    +0.7143   ✅
r2_deltactrl    drf_interpolated +0.4571    +0.4571   ✅
r2_deltactrl    drf_ctrl         +0.4571    NaN       ❌
```

即 `drf_ctrl` 在 5 个 `deltactrl` 指标上整列丢失。

另外两点语义提示（非 bug，但影响复现）：

- 原版 DRF 的"被评估对象"是 technical/interpolated **duplicate**（用于刻画指标的动态范围），port 返回的是"每个 method 相对基线的位置"，原版数值需取 `*_ceiling` 列。README 未说明这一点。
- 原版按 perturbation 聚合（median/mean/IQR，见 `calibration_analysis.py:553-600`），port 只输出 per-task 明细，聚合需用户自行完成。
- 边角：`DELTAPERT_METRIC_PREFIXES`（`metrics/_calibration.py:32`）把 `knn_jaccard_deltapert` 也映射到 control 基线，而原版 `CONTROL_BASELINE_METRICS` 不含该项；不过 `METRICS_CONFIG` 本来也不给 knn-jaccard 算 DRF，所以只是潜在不一致，不是当前结果的偏差。

### 3.3 P1：`Counterfactual.cellsimbench()` 覆盖面缺口

位置：`src/scdice_metrics/benchmark/_counterfactual.py:457`。

工厂实际启用的指标：`pearson_deltactrl`、`pearson_deltapert`、`r2_deltactrl`、`r2_deltapert`、`weighted_r2_deltactrl`、`weighted_r2_deltapert`、`mse_degs`、`wmse`，外加字段默认值带上的 `knn_jaccard_deltapert`。

与原版 `METRICS_CONFIG`（`calibration_analysis.py:56-79`）相比缺失：

- `nir`、`pds`（panel 指标，默认 `False`，工厂未开启）
- `mse`、`mae`（全谱版；`profile_mse`/`profile_mae` 已实现，但 `METRIC_FUNCTIONS` 未注册；`_calibration.METRIC_DIRECTION` 里却留了 `mse`/`mae` 条目）
- `mae_degs`、`wmae`
- `pearson_deltactrl_degs`、`pearson_deltapert_degs`、`r2_deltactrl_degs`、`r2_deltapert_degs`
- `pathway_recovery_deltapert`

按 `METRICS_CONFIG`（校准表，`calibration_analysis.py:56-79`）归类，缺口是上面的前四条；`pathway_recovery_deltapert` 不在 `METRICS_CONFIG` 中，它是**完整 `calculate_all_metrics`**（`metrics_engine.py`）才有、而 port 未实现的指标（同理 `deg_recovery_deltapert` 原版已注释）。反过来，`knn_jaccard_deltapert` 是 port 因字段默认值多出来的指标，校准表并不含它。

因此**使用 `Counterfactual.cellsimbench()` 跑不出 CellSimBench 论文的完整指标表**，只能得到其子集。

### 3.4 使用上的两个坑

1. `nir`/`pds` 的分组取自 `task.metadata["covariate"]`（`benchmark/_counterfactual.py` 的 `_covariate_groups`）。未设置时退化为 `"_all"` 全局计算，而原版严格**按 covariate 分组**（`metrics_engine.py:391-400`）。设置错误即无法复现。
2. `*deltapert*` 依赖调用方传入的 `task.dataset_mean`（原版为 fold 级 dataset-mean baseline），`*deltactrl*` 依赖 `task.reference`（原版为 donor/covariate 级 control baseline）。port 只做形状/长度校验，没有语义校验；baseline 传错数值就对不上。

---

## 4. 复现性结论

**当前可以直接对上原版的**：

`gt_mixture_entropy`、`spatial_ari`、`spatial_nmi`、`calinski_harabasz`、`davies_bouldin`、`cluster_specific_silhouette`(mean)、CellSimBench 的 `deg_*`、`mse/mae/wmse/wmae`、`r2_delta`、`pearson_*`、`nir`、`pds`、`knn_jaccard_deltapert`、`cross_method_ari`、DRF 的 `drf_mean`/`drf_interpolated`（以及非 `deltactrl` 指标的 `drf_ctrl`）。

**当前对不上原版的**：

`matched_mcc`、`matched_jaccard`（严重）、`domain_specific_f1`（`n_gt > n_clust` 的 NA 语义）、`pas`（默认标准化 + 近似 kNN）、`chaos`（ddof + 小簇规则）、`smoothness_entropy`（对数底数，数值约 1.4427 倍）、DRF `drf_ctrl`（`deltactrl` 指标）、`spatial_external_ari` / FMI / V_measure / LISI / `pathway_recovery`（缺失）。

---

## 5. 修复清单

### P0

1. `metrics/_spatial_supervised.py:139`、`:148`：不要分别 `factorize`。改为共享编码空间（例如把 `labels_arr` 与 `matched` 拼接后统一编码，或把匹配结果映射回 GT 的 code 空间），并显式决定未匹配簇是 NaN 还是保留为新类；补一个 golden 测试（可直接固化原版 `MCC.py`/`jaccard.py` 的输出，不依赖 R 运行时）。
2. `metrics/_calibration.py:56`：为 `*deltactrl*` 增加 mean 基线特例，使 `drf_ctrl` 与原版一致。

### P1

3. `metrics/_spatial_clustering.py`：PAS 默认改为 `standardize=False`（或在 `SpatialClustering.saccelerator()` 中显式关闭）；CHAOS 改用 ddof=1 的 z-score、只跳过单细胞簇；评估 PAS 是否改用精确 kNN。
4. `domain_specific_f1`：保留未被预测到的 domain（`f1=0` 或 NaN）并明确 `macro` 定义。
5. CellSimBench 覆盖面：注册 `mse`/`mae`；让 `cellsimbench()` 打开 `nir`/`pds`/`*_degs`/`mae*`/`wmae`；确认是否需要 GSEA `pathway_recovery_deltapert`。

### P2

6. `benchmark/_consensus.py:41`：改用自然对数以对齐原版。**邻域不用改**——两侧都排除自身、k=6（已用 dbscan 1.1-12 实测确认）。
7. 修正 `openspec/changes/add-saccelerator-metrics/tasks.md` 中不成立的 "PAS/CHAOS numerical alignment tests" 勾选，并补充 golden 测试目录（例如 `tests/golden/`）。
8. README 中补充：DRF 原版数值对应 `*_ceiling` 列；`cellsimbench()` 是子集而非全集；`knn_jaccard_deltapert` 不在原版校准表内。

---

## 附录 A：SACCELERATOR 对拍脚本

```python
# 运行环境：/home/chenxufeng/miniconda3/envs/systema/bin/python
# 依赖：numpy pandas scipy scikit-learn；R 侧需 clue/jsonlite
import importlib.util, sys, types, json, os, subprocess
import numpy as np, pandas as pd

ROOT = "/home/chenxufeng/WorkSpace/SpaPert/scDICE-metrics/src/scdice_metrics"
SAC = "/home/chenxufeng/WorkSpace/SpaPert/tmp/SACCELERATOR/metric"
PYT = "/home/chenxufeng/miniconda3/envs/systema/bin/python"
R_LIBS = "/home/chenxufeng/R/x86_64-conda-linux-gnu-library/4.3"


def load(name, path, stub=False, attrs=None):
    m = types.ModuleType(name)
    if stub:
        for k, v in (attrs or {}).items():
            setattr(m, k, v)
    sys.modules[name] = m
    if not stub:
        importlib.util.spec_from_file_location(name, path).loader.exec_module(m)
    return m


# 只加载被测模块；用 stub 切断 scdice_metrics 的 __init__ 重依赖
pkg = load("scdice_metrics", None, stub=True); pkg.__path__ = [ROOT]
mets = load("scdice_metrics.metrics", None, stub=True); mets.__path__ = [ROOT + "/metrics"]
load("scdice_metrics.nearest_neighbors", None, stub=True,
     attrs={"NeighborsResults": type("N", (), {}), "pynndescent": None})
load("scdice_metrics.utils", None, stub=True, attrs={"KMeans": None})
load("scdice_metrics.metrics._nmi_ari", ROOT + "/metrics/_nmi_ari.py")
sup = load("scdice_metrics.metrics._spatial_supervised", ROOT + "/metrics/_spatial_supervised.py")

D = "/tmp/sac_verify"
os.makedirs(D, exist_ok=True)


def write(gt, pred):
    idx = [f"s{i}" for i in range(len(gt))]
    pd.DataFrame({"label": gt}, index=idx).to_csv(f"{D}/gt.tsv", sep="\t")
    pd.DataFrame({"label": pred}, index=idx).to_csv(f"{D}/domains.tsv", sep="\t")


def orig(script, out):
    r = subprocess.run([PYT, f"{SAC}/{script}", "-l", f"{D}/domains.tsv",
                        "-g", f"{D}/gt.tsv", "-o", f"{D}/{out}"],
                       capture_output=True, text=True)
    return None if r.returncode != 0 else open(f"{D}/{out}").read().strip()


# --- 对抗例：匹配后标签的首次出现顺序与 GT 不同 ---
gt = np.array(["D0"] * 40 + ["D1"] * 100 + ["D0"] * 100)
pred = np.array(["X"] * 140 + ["Y"] * 100)
write(gt, pred)
print("MCC     orig:", orig("MCC/MCC.py", "m.txt"), " port:", sup.matched_mcc(gt, pred))
print("Jaccard orig:", json.loads(orig("jaccard/jaccard.py", "j.txt"))[0]["jaccard_score"],
      " port:", sup.matched_jaccard(gt, pred))
print("Entropy orig:", orig("Entropy/Entropy.py", "e.txt"),
      " port:", sup.gt_mixture_entropy(gt, pred))

# --- 随机场景统计 ---
rng = np.random.default_rng(1)
bad = 0
for trial in range(12):
    k = int(rng.integers(3, 8)); n = k * 80
    gl = [f"domain{chr(65 + i)}" for i in range(k)]
    gt = rng.choice(gl, size=n)
    cl = [f"clu{i}" for i in range(k)]; rng.shuffle(cl)
    m_ = dict(zip(gl, cl))
    pred = np.array([m_[g] for g in gt], dtype=object)
    nz = rng.random(n) < 0.25
    pred[nz] = rng.choice(cl, size=int(nz.sum()))
    write(gt, pred)
    mo = float(orig("MCC/MCC.py", "m.txt"))
    jo = float(json.loads(orig("jaccard/jaccard.py", "j.txt"))[0]["jaccard_score"])
    sm, sj = sup.matched_mcc(gt, pred), sup.matched_jaccard(gt, pred)
    if abs(mo - sm) > 1e-3 or abs(jo - sj) > 1e-3:
        bad += 1
        print(f"trial {trial}: MCC {mo:+.4f} vs {sm:+.4f} | Jacc {jo:.4f} vs {sj:.4f}")
print(f"mismatches: {bad}/12")

# --- domain-specific-f1（需 R）---
env = dict(os.environ, R_LIBS_USER=R_LIBS)
gt = rng.choice([f"domain{chr(65+i)}" for i in range(5)], size=500)
mp = {"domainA": "c1", "domainB": "c1", "domainC": "c2", "domainD": "c2", "domainE": "c3"}
pred = np.array([mp[g] for g in gt], dtype=object)
write(gt, pred)
subprocess.run(["Rscript", f"{SAC}/domain-specific-f1/domain-specific-f1.r",
                "-l", f"{D}/domains.tsv", "-g", f"{D}/gt.tsv", "-o", f"{D}/f1.json"], env=env)
print("R:", json.load(open(f"{D}/f1.json")))          # 未匹配 domain 为 NA
print("port:", sup.domain_specific_f1(gt, pred))       # 只含被匹配 domain
```

## 附录 B：CellSimBench 与 DRF 对拍脚本

```python
# 运行环境：/home/chenxufeng/miniconda3/envs/systema/bin/python
import importlib.util, sys, types
import numpy as np, pandas as pd

CSB = "/home/chenxufeng/WorkSpace/SpaPert/tmp/Perturbation-Models-Outperform-Baselines/cellsimbench"
ROOT = "/home/chenxufeng/WorkSpace/SpaPert/scDICE-metrics/src/scdice_metrics"

# 注入 stub 后加载原版实现（scanpy 在 module-level 被引用）
sc = types.ModuleType("scanpy"); sc.AnnData = type("AnnData", (), {})
sys.modules.setdefault("scanpy", sc)
pkg = types.ModuleType("cellsimbench"); pkg.__path__ = [CSB]; sys.modules["cellsimbench"] = pkg
core = types.ModuleType("cellsimbench.core"); core.__path__ = [CSB + "/core"]; sys.modules["cellsimbench.core"] = core
dm = types.ModuleType("cellsimbench.core.data_manager"); sys.modules["cellsimbench.core.data_manager"] = dm
importlib.util.spec_from_file_location("cellsimbench.core.data_manager", CSB + "/core/data_manager.py").loader.exec_module(dm)
me = types.ModuleType("cellsimbench.core.metrics_engine"); sys.modules["cellsimbench.core.metrics_engine"] = me
importlib.util.spec_from_file_location("cellsimbench.core.metrics_engine", CSB + "/core/metrics_engine.py").loader.exec_module(me)


def load(name, path, stub=False, attrs=None):
    m = types.ModuleType(name)
    if stub:
        for k, v in (attrs or {}).items():
            setattr(m, k, v)
    sys.modules[name] = m
    if not stub:
        importlib.util.spec_from_file_location(name, path).loader.exec_module(m)
    return m


spkg = load("scdice_metrics", None, stub=True); spkg.__path__ = [ROOT]
smets = load("scdice_metrics.metrics", None, stub=True); smets.__path__ = [ROOT + "/metrics"]
load("scdice_metrics.nearest_neighbors", None, stub=True, attrs={"NeighborsResults": type("N", (), {})})
load("scdice_metrics.utils", None, stub=True, attrs={"KMeans": None})
load("scdice_metrics.metrics._nmi_ari", ROOT + "/metrics/_nmi_ari.py")
load("scdice_metrics.metrics._perturbation", ROOT + "/metrics/_perturbation.py")
load("scdice_metrics.metrics._counterfactual", ROOT + "/metrics/_counterfactual.py")
csb = load("scdice_metrics.metrics._cellsimbench", ROOT + "/metrics/_cellsimbench.py")
cal = load("scdice_metrics.metrics._calibration", ROOT + "/metrics/_calibration.py")

rng = np.random.default_rng(0)
genes = [f"g{i}" for i in range(30)]

# --- DEG 权重/掩码 ---
deg_genes = [f"g{i}" for i in rng.integers(0, 30, size=60)]
scores = rng.normal(size=60); pvals = rng.random(60) * 0.2
Dm = object.__new__(dm.DataManager)
Dm.deg_names_dict = {"d1_p1": np.array(deg_genes)}
Dm.deg_scores_dict = {"d1_p1": scores}
Dm.deg_pvals_dict = {"d1_p1": pvals}
abs_s = np.abs(scores)
norm = np.square(np.nan_to_num((abs_s - abs_s.min()) / (abs_s.max() - abs_s.min())))
Dm.pert_normalized_abs_scores_vsrest_df = {
    "d1_p1": pd.DataFrame({"gene": deg_genes, "weight": norm}).groupby("gene")["weight"].max()
}
Dm.pert_normalized_abs_scores_vsrest = {"d1_p1": Dm.pert_normalized_abs_scores_vsrest_df["d1_p1"].values}
print("weights identical:",
      np.allclose(Dm.get_deg_weights("d1", "p1", gene_order=genes),
                  csb.deg_weights_from_scores(genes, scores, deg_genes)))
print("mask identical:",
      np.array_equal(Dm.get_deg_mask("d1", "p1", gene_order=genes, topn=100),
                     csb.deg_mask_from_pvals(genes, pvals, deg_genes, topn=100)))

# --- profile / delta 指标 ---
x1, x2, w = rng.normal(size=50), rng.normal(size=50), rng.random(50)
print(dm.mse(x1, x2), csb.profile_mse(x1, x2))
print(dm.wmse(x1, x2, w), csb.profile_wmse(x1, x2, w))
dt, dp = rng.normal(size=80), rng.normal(size=80)
print(dm.r2_score_on_deltas(dt, dp, weights=rng.random(80) * 50),
      csb.r2_delta(dt, dp, weights=rng.random(80) * 50))   # 注意两侧权重需一致

# --- NIR / KNN-Jaccard ---
keys = [f"d1_p{i}" for i in range(6)]
P = pd.DataFrame(rng.normal(size=(6, 20)), index=keys)
T = pd.DataFrame(rng.normal(size=(6, 20)), index=keys)
E = object.__new__(me.MetricsEngine)
print(E._calculate_nir_scores(P, T)["d1_p2"],
      csb.nir_scores(P, T, covariate_groups={"d1": keys})["d1_p2"])
print(E._calculate_knn_jaccard_scores(P, T, k=3)["d1_p2"],
      csb.knn_jaccard_deltapert(P, T, k=3)["d1_p2"])

# --- DRF：deltactrl 指标的 drf_ctrl 分歧 ---
vals = {
    "r2_deltapert": {"dataset_mean": 0.05, "control": 0.20, "technical_duplicate": 0.72,
                     "interpolated_duplicate": 0.55, "model": 0.61},
    "r2_deltactrl": {"dataset_mean": 0.30, "control": np.nan, "technical_duplicate": 0.80,
                     "interpolated_duplicate": 0.62, "model": 0.68},
}
rows = [{"task_id": "t1", "method": m, "metric": k, "value": v}
        for k, d in vals.items() for m, v in d.items()]
out = cal.drf_all_versions(pd.DataFrame(rows))
print(out[out.metric == "r2_deltactrl"][
    ["method", "null_baseline_method", "drf_ctrl", "drf_ctrl_ceiling"]].to_string(index=False))
# 原版 drf_ctrl = +0.4571（deltactrl 指标改用 dataset_mean 作基线）
```
