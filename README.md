# Task 1 — Au20 ΔE Prediction (SOAP+KRR Baselines, SchNet GNN, KD/VKD, Conformal Prediction)

> 项目路径：`D:\DataComp\DataCompetition`
> Python 解释器：`D:\DataComp\DataCompetition\.conda_env\python.exe`
> 数据：999 个 `Au20` 等原子数簇的 `.xyz`（固定测试集 155 个样本）

本仓库包含 Task 1 的完整实现：

* 经典 **SOAP + KRR** 基线（含中心化目标 & L2 特征归一化、异常样本降权、固定折评估、集成评测）；
* 轻量 **SchNet 风格 GNN**（支持蒸馏 *KD* 与邻域蒸馏 *Vicinal KD*，GPU 训练）；
* **Conformal Prediction (CP)** 不确定性评估（全局与 Mondrian 分组）；
* 可复现实验与可视化产物（metrics、预测、图表）。

---

## 目录结构（关键路径）

```
DataCompetition/
├─ data/
│  ├─ raw/Au20_OPT_1000/              # 999 × .xyz
│  └─ preproc_out/                    # 预处理输出
│     ├─ energies_nohash.csv
│     ├─ summary.json
│     ├─ cv_folds_fixed.json          # 固定5折
│     └─ test_ids_nohash.json         # 固定测试ID（155）
├─ models/
│  ├─ baselines/soap_krr/
│  │  ├─ train_fixedsplits.py         # 训练（固定折）
│  │  ├─ eval_fixedtest.py            # 测试集评估
│  │  ├─ tools_check_id_alignment.py  # ID/能量/几何对齐检查
│  │  ├─ tools_make_folds.py          # 生成固定5折
│  │  ├─ eval_ensemble.py             # 多run平均评测
│  │  ├─ features.py, dataset.py, model.py, plots.py
│  └─ gnn/schnet/
│     ├─ train_gnn_schnet.py          # SchNet 训练（支持 KD/VKD）
│     ├─ eval_gnn_fixedtest.py        # SchNet 测试评估
│     └─ eval_gnn_conformal.py        # SchNet + CP 评估
└─ runs/
   ├─ soap_krr_run2_centered_l2/
   ├─ soap_krr_run3_rcut6_n10_l8_s0p5_w04/
   ├─ soap_krr_run4_rcut5p5_n8_l8_s0p3_mean_w04/   # 最优KRR单模（示例）
   ├─ gnn_schnet_kd/fold1_batched/                 # GNN KD-only（最佳）
   └─ gnn_schnet_kd_vkd/fold1/                     # GNN KD+VKD（可选）
```

---

## 0) 环境准备


```bash
# Git Bash 下示例（Windows CMD 的换行符需要用 ^，见下文提示）
PYTHONNOUSERSITE=1 "/d/DataComp/DataCompetition/.conda_env/python.exe" -m pip install -U \
  numpy pandas scipy scikit-learn ase dscribe matplotlib joblib tqdm packaging typing_extensions

# 有 CUDA 时建议装对应 PyTorch（如 cu121）
PYTHONNOUSERSITE=1 "/d/DataComp/DataCompetition/.conda_env/python.exe" -m pip install -U \
  torch --index-url https://download.pytorch.org/whl/cu121
```

## 1) 数据放置

* `.xyz` 原始几何：`D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000`
* 预处理产物：`D:\DataComp\DataCompetition\data\preproc_out\`

  * `energies_nohash.csv`、`summary.json`、`cv_folds_fixed.json`、`test_ids_nohash.json`

如需重建固定 5 折，可运行：

```bash
cd /d/DataComp/DataCompetition
PYTHONNOUSERSITE=1 "./.conda_env/python.exe" \
  models/baselines/soap_krr/tools_make_folds.py \
  --energies_csv "D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv" \
  --xyz_dir     "D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000" \
  --test_ids_json "D:\DataComp\DataCompetition\data\preproc_out\test_ids_nohash.json" \
  --out_folds_json "D:\DataComp\DataCompetition\data\preproc_out\cv_folds_fixed.json"
```

---

## 2) 基线：SOAP + KRR

### 2.1 训练（固定折交叉验证选 α）

```bash
cd /d/DataComp/DataCompetition/models/baselines/soap_krr
PYTHONNOUSERSITE=1 "/d/DataComp/DataCompetition/.conda_env/python.exe" train_fixedsplits.py \
  --energies_csv "D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv" \
  --summary_json "D:\DataComp\DataCompetition\data\preproc_out\summary.json" \
  --cv_folds_json "D:\DataComp\DataCompetition\data\preproc_out\cv_folds_fixed.json" \
  --test_ids_json "D:\DataComp\DataCompetition\data\preproc_out\test_ids_nohash.json" \
  --xyz_root "D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000" \
  --source_mode dir --id_mode filename \
  --rcut 5.5 --nmax 8 --lmax 8 --sigma 0.3 --pooling mean \
  --alpha_grid 1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 \
  --use_outlier_weights --outlier_weight 0.4 \
  --out_dir "D:\DataComp\DataCompetition\runs\soap_krr_run4_rcut5p5_n8_l8_s0p3_mean_w04"
```

> **说明**
>
> * 已在管线中实现：目标**中心化**、特征 **L2** 归一化；
> * `--use_outlier_weights` 会对 `is_outlier==1` 的样本传入较小权重到 `KRR.fit(sample_weight=...)`；
> * 5 折网格搜索选 α，最终会保存 `model.joblib` 与 `meta.json`。

### 2.2 固定测试集评估

```bash
cd /d/DataComp/DataCompetition/models/baselines/soap_krr
PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONIOENCODING=UTF-8 "/d/DataComp/DataCompetition/.conda_env/python.exe" \
  eval_fixedtest.py \
  --model_path "D:\DataComp\DataCompetition\runs\soap_krr_run4_rcut5p5_n8_l8_s0p3_mean_w04\model.joblib" \
  --meta_path  "D:\DataComp\DataCompetition\runs\soap_krr_run4_rcut5p5_n8_l8_s0p3_mean_w04\meta.json" \
  --energies_csv "D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv" \
  --test_ids_json "D:\DataComp\DataCompetition\data\preproc_out\test_ids_nohash.json" \
  --xyz_root "D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000" \
  --source_mode dir --id_mode filename \
  --out_dir "D:\DataComp\DataCompetition\runs\soap_krr_run4_rcut5p5_n8_l8_s0p3_mean_w04"
```

> 输出：`metrics.json`、`test_predictions.csv`。

### 2.3 多模型平均（可选）

把多个 run 的 `run_dir` 传给 `eval_ensemble.py` 做简单平均：

```bash
cd /d/DataComp/DataCompetition/models/baselines/soap_krr
PYTHONNOUSERSITE=1 "/d/DataComp/DataCompetition/.conda_env/python.exe" \
  eval_ensemble.py \
  --run_dirs "D:\DataComp\DataCompetition\runs\soap_krr_run2_centered_l2" \
            "D:\DataComp\DataCompetition\runs\soap_krr_run3_rcut6_n10_l8_s0p5_w04" \
            "D:\DataComp\DataCompetition\runs\soap_krr_run4_rcut5p5_n8_l8_s0p3_mean_w04" \
  --energies_csv "D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv" \
  --test_ids_json "D:\DataComp\DataCompetition\data\preproc_out\test_ids_nohash.json" \
  --xyz_root "D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000" \
  --source_mode dir --id_mode filename \
  --out_dir "D:\DataComp\DataCompetition\runs\soap_krr_ensemble_r2r3r4"
```

---

## 3) SchNet 风格 GNN（支持 KD / VKD）

> 已在 `models/gnn/schnet/train_gnn_schnet.py` 中集成了：
>
> * **批量前向**、**AMP**、**早停**、**ReduceLROnPlateau**、**异常降权 w**；
> * **KD**：从 KRR 老师（推荐 run4）蒸馏软标；
> * **VKD（可选）**：训练时对坐标做小扰动并用老师软标监督邻域。

### 3.1 训练（KD-only，fold=1）

```bash
cd /d/DataComp/DataCompetition

PYTHONNOUSERSITE=1 "/d/DataComp/DataCompetition/.conda_env/python.exe" \
  models/gnn/schnet/train_gnn_schnet.py \
  --energies_csv "D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv" \
  --cv_folds_json "D:\DataComp\DataCompetition\data\preproc_out\cv_folds_fixed.json" \
  --test_ids_json "D:\DataComp\DataCompetition\data\preproc_out\test_ids_nohash.json" \
  --xyz_dir "D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000" \
  --fold 1 \
  --out_dir "D:\DataComp\DataCompetition\runs\gnn_schnet_kd\fold1_batched" \
  --hidden_dim 128 --n_blocks 4 --n_rbf 64 --rcut 6.0 \
  --batch_size 256 --lr 2e-3 --weight_decay 1e-4 --epochs 800 --patience 120 \
  --use_outlier_weights --outlier_weight 0.6 \
  --teacher_dir "D:\DataComp\DataCompetition\runs\soap_krr_run4_rcut5p5_n8_l8_s0p3_mean_w04" \
  --lambda_kd 0.3 \
  --device cuda
```

### 3.2 测试评估（GNN）

```bash
PYTHONNOUSERSITE=1 "/d/DataComp/DataCompetition/.conda_env/python.exe" \
  models/gnn/schnet/eval_gnn_fixedtest.py \
  --energies_csv "D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv" \
  --test_ids_json "D:\DataComp\DataCompetition\data\preproc_out\test_ids_nohash.json" \
  --xyz_dir "D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000" \
  --model_path "D:\DataComp\DataCompetition\runs\gnn_schnet_kd\fold1_batched\model_fold1.pt" \
  --hidden_dim 128 --n_blocks 4 --n_rbf 64 --rcut 6.0 \
  --out_dir "D:\DataComp\DataCompetition\runs\gnn_schnet_kd\fold1_batched" \
  --device cuda
```

> 得到的代表性结果（fold1，KD-only）：
> `MAE=0.4386, RMSE=0.6465, R²=0.9509`（显著优于 KRR 基线）

---

## 4) Conformal Prediction (CP) 不确定性评估

**全局 CP：**

```bash
PYTHONNOUSERSITE=1 "/d/DataComp/DataCompetition/.conda_env/python.exe" \
  models/gnn/schnet/eval_gnn_conformal.py \
  --energies_csv "D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv" \
  --cv_folds_json "D:\DataComp\DataCompetition\data\preproc_out\cv_folds_fixed.json" \
  --fold 1 \
  --test_ids_json "D:\DataComp\DataCompetition\data\preproc_out\test_ids_nohash.json" \
  --xyz_dir "D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000" \
  --model_path "D:\DataComp\DataCompetition\runs\gnn_schnet_kd\fold1_batched\model_fold1.pt" \
  --hidden_dim 128 --n_blocks 4 --n_rbf 64 --rcut 6.0 \
  --batch_size 256 \
  --alphas 0.1 0.2 \
  --out_dir "D:\DataComp\DataCompetition\runs\gnn_schnet_kd\fold1_batched" \
  --device cuda
```

**Mondrian（按能量桶分组）CP：**

```bash
PYTHONNOUSERSITE=1 "/d/DataComp/DataCompetition/.conda_env/python.exe" \
  models/gnn/schnet/eval_gnn_conformal.py \
  --energies_csv "D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv" \
  --cv_folds_json "D:\DataComp\DataCompetition\data\preproc_out\cv_folds_fixed.json" \
  --fold 1 \
  --test_ids_json "D:\DataComp\DataCompetition\data\preproc_out\test_ids_nohash.json" \
  --xyz_dir "D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000" \
  --model_path "D:\DataComp\DataCompetition\runs\gnn_schnet_kd\fold1_batched\model_fold1.pt" \
  --hidden_dim 128 --n_blocks 4 --n_rbf 64 --rcut 6.0 \
  --batch_size 256 \
  --alphas 0.1 0.2 \
  --mondrian \
  --out_dir "D:\DataComp\DataCompetition\runs\gnn_schnet_kd\fold1_batched" \
  --device cuda
```

> 代表性输出（fold1, KD-only，全局 CP，α=0.1/0.2）
> `PICP` 接近 `1-α`、`MPIW` 合理；`q_by_bucket` 显示高能桶更宽，符合物理直觉。

---

## 5) 结果总览（示例）

| 模型 / 设定            | MAE       | RMSE      | R²        |
| ------------------ | --------- | --------- | --------- |
| KRR run2（中心化+L2）   | 0.726     | 0.978     | 0.888     |
| KRR run3（多尺度+降权）   | 0.724     | 1.051     | 0.870     |
| KRR run4（最优单模）     | **0.669** | **0.946** | **0.895** |
| GNN（KD-only，fold1） | **0.439** | **0.647** | **0.951** |

> 结论：GNN（KD）相对最优 KRR 单模 **MAE 降约 34%**、**RMSE 降约 32%**，R² 明显提升。
> CP 在 α=0.1/0.2 下达到目标覆盖率，区间宽度随能量桶上升而增大。

---

## 6) 复现与可视化

* **随机种子**：在 `train_gnn_schnet.py`/`eval_*` 内固定 `numpy/random/torch` 种子（已提供/可选项）。
* **版本**：Python 3.11；`numpy 2.3.x / pandas 2.2.x / scipy 1.16.x / scikit-learn 1.5.x / ase 3.26 / dscribe 2.1.x / torch 2.5.x+cu121`。
* **硬件**：CUDA 12.5，单卡训练；KRR 特征生成在 CPU，GNN 训练/推理在 GPU。
* **图表**：评估脚本会输出 `parity.png`、`residuals_by_bucket.png`、`cp_calibration.json`（具体以脚本为准）。如需论文风格图，请在 `runs/...` 中调用 `plots.py` 或你自己的绘图脚本。

---

## 7) 常见问题（FAQ）

* **Unicode 输出报错**：在命令前加 `PYTHONUTF8=1 PYTHONIOENCODING=UTF-8`。
* **`ModuleNotFoundError: packaging/typing_extensions`**：`pip install -U packaging typing_extensions`。
* **`index_add_(): self(Float) and source(Half)`**：已在 GNN 中屏蔽了 AMP 对聚合的混精影响（强制 `float32` 聚合）。
* **`dscribe.SOAP` API 差异**：已适配当前版本（使用 `r_cut`/`n_max`/`l_max`/`sigma` 等实际参数名）。
* **Windows CMD 与 Git Bash**：CMD 的多行续行用 `^`，Bash 用 `\`；路径分隔符可用双引号包裹的 Windows 路径。

---

## 8) 设计要点（论文写作摘要）

* **SOAP+KRR**：局域电子密度 → 球谐/径向投影 → 旋转不变功率谱；结构级向量用 sum/mean 池化（能量可加性）。小样本下强基线；中心化目标与 L2 特征归一化稳定收敛；异常样本降权进一步降低 RMSE。
* **SchNet**：元素嵌入 + 连续滤波卷积（径向核，仅用距离，E(3)-不变）+ 残差堆叠 → 原子能读出求和（可加性）。**KD** 用 KRR 软标稳住训练；**VKD** 通过几何邻域扰动的软标逼近能量流形，增强泛化。
* **CP**：在固定验证集校准分位数，测试时输出覆盖有保证的区间；Mondrian 版对不同能量桶自适应区间宽度，解释性更好。

---

### Windows CMD 写法（示例）

把上文任一 Bash 命令改为：

```bat
SET PYTHONNOUSERSITE=1
"D:\DataComp\DataCompetition\.conda_env\python.exe" ^
  models\gnn\schnet\train_gnn_schnet.py ^
  --energies_csv "D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv" ^
  --cv_folds_json "D:\DataComp\DataCompetition\data\preproc_out\cv_folds_fixed.json" ^
  --test_ids_json "D:\DataComp\DataCompetition\data\preproc_out\test_ids_nohash.json" ^
  --xyz_dir "D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000" ^
  --fold 1 ^
  --out_dir "D:\DataComp\DataCompetition\runs\gnn_schnet_kd\fold1_batched" ^
  --hidden_dim 128 --n_blocks 4 --n_rbf 64 --rcut 6.0 ^
  --batch_size 256 --lr 2e-3 --weight_decay 1e-4 --epochs 800 --patience 120 ^
  --use_outlier_weights --outlier_weight 0.6 ^
  --teacher_dir "D:\DataComp\DataCompetition\runs\soap_krr_run4_rcut5p5_n8_l8_s0p3_mean_w04" ^
  --lambda_kd 0.3 ^
  --device cuda
```

---

如需我把 README 直接改成英文版、或把你的 **图表/表格自动化绘制**脚本也写入 README（一步出图），告诉我你想要的风格就行。
