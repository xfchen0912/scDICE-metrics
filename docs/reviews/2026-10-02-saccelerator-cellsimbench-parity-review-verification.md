# 复核：SACCELERATOR / CellSimBench 原版一致性审查

- 日期：2026-10-02
- 被复核文档：`docs/reviews/2026-10-02-saccelerator-cellsimbench-parity-review.md`
- 代码对象：`scDICE-metrics` HEAD `02fb1b2`（与原审查相同）
  - 其前一提交 `4bfdea4`（CellSimBench-style）
- 上游 SHA 复核：SACCELERATOR `dac468d`；CellSimBench `15361ae`（与原审查相同）
- 方法：源码对照 + 独立数值复现（不依赖原审查附录脚本原样重放）。Python 为 SpaPert-model_factory pixi env（sklearn 1.7.2 / pandas 2.3.3 / numpy 1.26.4）。`dbscan::kNN` 的自身邻域行为以 dbscan 源码与 CRAN 文档为准（本机未装 `dbscan`）。未重跑原版 R `PAS.r` / `CHAOS.r` / `domain-specific-f1.r`，也未重新注入 `cellsimbench.core` 做全套对拍。

---

## 0. 结论摘要

原审查的核心结论成立：`matched_mcc` / `matched_jaccard` 和 DRF `drf_ctrl` 是真实 P0；PAS/CHAOS、`domain_specific_f1`、`cellsimbench()` 覆盖面是真实偏差；matched 系列当前不能当作 SACCELERATOR 数值替代。

**后续（同日）**：PAS/CHAOS 已按 SDMBench（非 SACCELERATOR R）对齐。默认 `pas(standardize=False, reuse_spatial_neighbors=False)` 走精确 kNN；`chaos` 保持 `StandardScaler` + 跳过 ≤2 细胞簇。`domain_specific_f1` 已按 SACCELERATOR 保留未匹配 domain 为 `NaN`，`macro` 为全部 domain 的均值（有未匹配则为 `NaN`）。`smoothness_entropy` 已改为自然对数。

有一处实质写错：`smoothness_entropy` 的邻域「包含自身」不成立（只有对数底数差异）。另有几处分类/机制表述需要收紧。

| 原审查条目 | 复核 |
| --- | --- |
| `matched_mcc` / `matched_jaccard` 编码错位（符号翻转、4/12 随机失败） | **确认**，对抗例与 12 次随机试验数值与原审查完全一致 |
| DRF `drf_ctrl` 在 5 个 `*deltactrl*` 指标上整列 NaN | **确认** |
| `domain_specific_f1` 在 `n_gt > n_clust` 时不表达未匹配 domain | **确认**（代码路径；未重跑 R） |
| PAS 默认 `standardize=True` + Benchmarker 近似 kNN | **确认** |
| CHAOS ddof=0 vs R `scale()` ddof=1；跳过 2 细胞簇 | **确认** |
| `cellsimbench()` 是子集；全谱 `mse`/`mae` 未注册 | **确认** |
| `spatial_external_ari` / FMI / V_measure / GSEA 缺失 | **确认** |
| 现有测试挡不住 MCC bug；`tasks.md` PAS/CHAOS 对齐勾选不实 | **确认** |
| `gt_mixture_entropy` 对抗例两侧同值 | **确认** `0.5034869983305348` |
| `smoothness_entropy` 邻域包含自身 | **不成立**；只有 `log` vs `log2` |
| 「matched 系列当前结果不可用」 | **成立**（不是每次都错，但是系统性地不可靠） |

---

## 1. 已独立复现的问题

### 1.1 P0：`matched_mcc` / `matched_jaccard` 标签编码错位

位置：`src/scdice_metrics/metrics/_spatial_supervised.py` 的 `matched_mcc` / `matched_jaccard`。

`match_predicted_to_ground_truth` 返回的是 GT **字符串标签**。随后：

```python
gt_codes, _ = pd.factorize(labels_arr)    # GT 数组的首次出现顺序
pred_codes, _ = pd.factorize(matched)     # 匹配后数组的首次出现顺序
```

原版 `MCC.py` / `jaccard.py` 是先 `.astype("category").cat.codes`，再在**整数编码空间**做匈牙利匹配；匹配后两边已经在 GT 的 code 空间里。port 错在匹配成字符串之后又各 `factorize` 一次。

对抗构造（与原审查相同：`gt = [D0]×40 + [D1]×100 + [D0]×100`，`pred = [X]×140 + [Y]×100`）：

| 指标 | 原版逻辑 | scDICE port |
| --- | --- | --- |
| MCC | **+0.714286** | **-0.714286**（符号翻转） |
| Jaccard | **0.714286** | **0.097222** |
| Entropy | 0.503487 | 0.503487 |

随机场景（k 个 domain、25% 误分、`n_clust == n_gt`，`rng` seed=1）12 例中 **4 例严重不符**，trial 编号与数值与原审查一致：

```
trial 4  k=7: MCC +0.7188 vs -0.1146 | Jaccard 0.6131 vs 0.0240
trial 5  k=5: MCC +0.7438 vs +0.1921 | Jaccard 0.6611 vs 0.2946
trial 6  k=4: MCC +0.7462 vs +0.2315 | Jaccard 0.6829 vs 0.3383
trial 8  k=5: MCC +0.8063 vs +0.2435 | Jaccard 0.7331 vs 0.3329
```

`tests/metrics/test_spatial_saccelerator.py` 的完美匹配用例（`labels == pred`）编码碰巧对齐，测不出该问题。

补充：`n_clust > n_gt` 时原版 `domains.map(dict(zip(row_ind, col_ind)))` 给未匹配簇写 `NaN`，`matthews_corrcoef` 会报错。port 保留未匹配簇原始标签。`workflows/04_metrics.smk` 未传 `--matched_labels`。这是健壮性差异，不能据此声称与原版数值一致。

机制表述修正：原审查写「原版按类别名排序，两边天然共享同一编码空间」。pandas 2.x 上 `astype("category")` 对字符串类别确实按名字排序（`["D1","D0"]` → categories `['D0','D1']`），这一点成立。真正让原版不翻车的，是**先编码再在整数空间做匈牙利**；不是两边一开始就共享编码。port 的 bug 是匹配成字符串后二次 `factorize`。结论不变。

### 1.2 P0：DRF `drf_ctrl` 基线选择错误

原版 `calibration_analysis.py` 的 `MEAN_BASELINE_METRICS_FOR_CTRL` 对 5 个 `*deltactrl*` 指标改用 mean 基线（control 预测相对 control 的 deltactrl 相关/R² 退化为 NaN）。

port 的 `_null_baseline_method_for_metric` 只把 `*deltapert*` 前缀映射到 `control`；但 `drf_all_versions()` 计算 `drf_ctrl` 时对**所有**指标强制 `null_baseline_method="control"`。

同一组数：`r2_deltactrl` 原版 `drf_ctrl` 应为 **+0.4571**，port 为 **NaN**。`drf_mean` / `drf_interpolated` 以及非 `deltactrl` 指标的 `drf_ctrl` 正确。

语义提示也成立：原版 DRF 的被评估对象是 technical/interpolated duplicate，对应 port 的 `*_ceiling` 列；`score_relative_to_baselines` 会跳过 baseline/ceiling method 本身。README 未说明这一点。

### 1.3 P1：`domain_specific_f1` 的 NA 语义

原版 `domain-specific-f1.r` 在 `n_gt > n_clust` 时用 `mapping[names(tt), "domain_size"] <- tt` 给未匹配 domain 补行，写出的 `f1` 为 NA。port 只遍历匈牙利匹配对，`macro` 只在被匹配 domain 上取均值。未匹配 GT domain 从输出消失，macro 偏乐观。

`n_clust == n_gt` 与 `n_clust > n_gt`（dummy 补列）的实现结构与 R 一致；本次未重跑 R 做逐值对拍。

### 1.4 P1：PAS / CHAOS 参数级偏差

**PAS**（`metrics/_spatial_clustering.py`）

- 原版 `PAS.r`：原始坐标 + 精确 10-NN（`pdist` 排除自身），`sum(neighbor_label != self) > k/2`。
- 直接 API 默认 `standardize=True`。各向异性坐标复现：`standardize=False` 与原版同值，`True` 改变近邻排序。
- `SpatialClustering.saccelerator()` 未显式关闭标准化。
- Benchmarker 路径：`build_spatial_neighbors` → `pynndescent` 近似 kNN，`k = max(spatial_n_neighbors, pas_k+1)`（默认 ≥11），坐标未经 z-score。这与直接 `pas()` 的标准化问题是两条路径。

**CHAOS**

- 原版 `scale()` 为样本标准差（ddof=1）；port `StandardScaler` 为 ddof=0。无小簇时比值精确等于 `sqrt(n/(n-1))`（n=400 时 1.001252，约 +0.13%）。
- 原版 `if(length(location_cluster)==2) next`：2D 坐标下单行被 drop 成长度 2 的向量，因此只跳过**单细胞簇**；2 细胞簇是 2×2 矩阵，`length=4`，不跳过。
- port 跳过 `shape[0] <= 2`。构造 2 细胞簇后，原版纳入、port 跳过；相对差取决于小簇 1-NN 距离，本次构造约 -14%。原审查给的 -1.6% 是其数据上的幅度，方向和机制成立。

`openspec/changes/add-saccelerator-metrics/tasks.md` 已勾选 “PAS/CHAOS numerical alignment tests vs SACCELERATOR reference”。仓库中只有有限性/排序 sanity（`test_pas_chaos_finite`、`test_chaos_prefers_spatially_compact_clusters`），没有相对原版的 golden。勾选不成立。

### 1.5 P1：`Counterfactual.cellsimbench()` 覆盖面

工厂实际启用：`pearson_deltactrl` / `pearson_deltapert` / `r2_deltactrl` / `r2_deltapert` / `weighted_r2_*` / `mse_degs` / `wmse`，外加 dataclass 默认值带来的 `knn_jaccard_deltapert={"k": 20}`。

相对原版 `calculate_all_metrics` / 校准 `METRICS_CONFIG`：

- `nir` / `pds`：默认 `False`，工厂未开
- 全谱 `mse` / `mae`：`profile_mse` / `profile_mae` 已实现，**未注册进 `METRIC_FUNCTIONS`**；`METRIC_DIRECTION` 里却有条目
- `mae_degs` / `wmae`：**已注册**，工厂未开
- `pearson_*_degs` / `r2_*_degs`：已注册，工厂未开
- `pathway_recovery_deltapert`：未实现

使用陷阱也成立：

1. `nir` / `pds` 分组取自 `task.metadata["covariate"]`，缺省为 `"_all"`；原版按 `pert_key.split('_')[0]` 强制分组。
2. `r2_delta` 权重全 0：原版退回无加权 R²，port 返回 NaN（已复现）。

### 1.6 缺失与已对齐项

确认缺失：`spatial_external_ari` 恒 NaN；FMI / V_measure 未实现；SACCELERATOR `LISI.r` 未 port（仓库 `lisi_knn` 是 scIB 风格）；GSEA 未实现。`deg_recovery_deltapert` 原版已注释，优先级低。

确认可对齐（本次复现或代码等价）：`gt_mixture_entropy` 对抗例同值；`spatial_ari` / `spatial_nmi`（纯标签、sklearn）；CH / DB（sklearn 直调）；`cluster_specific_silhouette` 出 per-cluster mean + 全体样本 mean（原版另有 median）；`cross_method_ari` 与 `mclust::adjustedRandIndex` 公式等价。

CellSimBench 原子函数（`deg_weights` / `deg_mask` / `profile_*` / `pearson_*` / `nir` / `pds` / `knn_jaccard`）本次只做代码对照，未重新注入原版 `cellsimbench.core`。从实现看原审查「逐元素一致」可信；「原子指标可直接替代」需加上 `r2_delta` 全零权重与 `drf_ctrl` 两条注脚。

---

## 2. 原审查需要修正的点

### 2.1 `smoothness_entropy` 邻域含自身 — 不成立

位置：`src/scdice_metrics/benchmark/_consensus.py` vs 原版 `Smoothness_entropy.r`。

**对数底数差异成立**：原版 `calc_entropy` 用自然对数 `log`，port 用 `log2`。倍数精确为 `1/ln2 ≈ 1.4427`。

**邻域构成差异不成立**。`dbscan::kNN` 默认排除自身（源码注释 `returns NO self matches`；CRAN：「If no query is specified, then self-matches are removed」）。`k=6` 是 6 个邻居，不含 query 点。port 的 `np.argsort(dist)[:, 1:k+1]` 同样排除自身、取 6 个邻居。

复现：port 与「排除自身 + log2」数值完全相同；与「包含自身 + ln」对不上。P2 修复只需改对数底数，不必改邻域。

### 2.2 `pathway_recovery` 不在校准 `METRICS_CONFIG` 里

`calibration_analysis.py:56-78` 的 `METRICS_CONFIG` 没有 GSEA，也没有 `knn_jaccard_deltapert`。把 `pathway_recovery_deltapert` 写进「相对 METRICS_CONFIG 的缺口」不准确；作为「相对完整 `calculate_all_metrics` 的缺口」才对。

反过来，`cellsimbench()` 因字段默认值启用了 `knn_jaccard_deltapert`，相对校准表是多出来的指标。

---

## 3. 原审查未写、但相关的边角

`_null_baseline_method_for_metric` 用 `startswith(DELTAPERT_METRIC_PREFIXES)`，因此 `knn_jaccard_deltapert` 的 `drf_mean` 会走 control 基线。原版 `CONTROL_BASELINE_METRICS` 不含该项，且校准 `METRICS_CONFIG` 不算它的 DRF。若用 port 给 knn-jaccard 算 DRF，会与「按 METRICS_CONFIG 默认规则」不一致。不是 P0，但是同一处基线逻辑的边角。

---

## 4. 修复清单复核

原审查 P0/P1/P2 优先级仍然合理，仅第 6 条需改写：

### P0（不变）

1. `matched_mcc` / `matched_jaccard`：不要分别 `factorize`。改为共享编码空间（拼接后统一编码，或把匹配结果映射回 GT code 空间），并显式决定未匹配簇是 NaN 还是保留为新类；补 golden 测试（可固化原版 `MCC.py` / `jaccard.py` 输出）。
2. `_calibration.py`：为 `*deltactrl*` 增加 mean 基线特例，使 `drf_ctrl` 与原版一致。

### P1（不变）

3. PAS 默认改为 `standardize=False`（或在 `SpatialClustering.saccelerator()` 中显式关闭）；CHAOS 改用 ddof=1、只跳过单细胞簇；评估 PAS 是否改用精确 kNN。
4. `domain_specific_f1`：保留未被预测到的 domain（`f1=0` 或 NaN）并明确 `macro` 定义。
5. 注册 `mse`/`mae`；让 `cellsimbench()` 打开 `nir`/`pds`/`*_degs`/`mae*`/`wmae`；确认是否需要 GSEA。

### P2（第 6 条改写）

6. `smoothness_entropy`：改用自然对数以对齐原版。**不必改邻域**（两侧都排除自身、k=6）。
7. 修正 `tasks.md` 中不成立的 PAS/CHAOS 对齐勾选，并补 golden。
8. README：DRF 原版数值对应 `*_ceiling` 列；`cellsimbench()` 是子集而非全集。
