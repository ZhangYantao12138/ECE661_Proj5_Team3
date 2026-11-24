# ECE661 Team 3 - Universal Adversarial Patches on CIFAR-10

Completed implementation and measured experiments for ECE661 Project 5, Fall 2026. The original [proposal](proposal.md) is preserved.

**全部实验已完成。** 三个分类器各完成 200 epochs；561 份主补丁、24 份独立强度检查；1,212 个完整源验证条件和 1,638 个正式测试条件通过逐样本审计。全部数值来自真实运行。七份 PDF 共 54 页经过渲染检查；四个代码包在新目录合并后通过 26 项测试和全部 585 份补丁身份检查。

## Downloads and reports

- [Release: code, trained artifacts, data and all raw evidence](https://github.com/ZhangYantao12138/ECE661_Proj5_Team3/releases/tag/v1.0.0-ece661)
- [Code and documents convenience bundle, about 50 MB](https://github.com/ZhangYantao12138/ECE661_Proj5_Team3/releases/download/v1.0.0-ece661/ECE661_Code_and_Documents.zip)
- [Formal report](reports/Formal_Report.pdf): six body pages plus references; official NeurIPS 2026 style, identified as an ECE661 course report.
- Chinese illustrated guides: [A](reports/chinese/A_Chinese_Guide.pdf), [B](reports/chinese/B_Chinese_Guide.pdf), [C](reports/chinese/C_Chinese_Guide.pdf), [D](reports/chinese/D_Chinese_Guide.pdf).
- [Supplementary results](reports/Supplementary_Results.pdf), [course poster](reports/ECE661_Poster.pdf), [LaTeX source](reports/latex/main.tex).

The complete 5.15 GB delivery is split into three byte-identical segments because GitHub limits individual release assets. Download all three `.partNN` files, `reassemble_delivery.py` and `release_assets.json` from the release into one directory. Then run:

```powershell
python reassemble_delivery.py --parts-dir . --output ECE661_Full_Deliverables.zip
```

The utility checks each segment and the reconstructed archive against SHA-256. The complete archive contains the independently packaged A/B/C/D code, trained models and patches, official CIFAR-10 archive, all per-image/per-draw records, logs, reports, source and original inputs. It is the authoritative full delivery. The 50 MB convenience ZIP excludes raw records and the dataset.

## Measured results

| Model | Clean test accuracy | Parameters |
|---|---:|---:|
| ResNet-20 | 92.29% | 269,722 |
| VGG-11-BN-CIFAR | 91.99% | 9,756,426 |
| DenseNet-BC-40-12 | 92.36% | 176,122 |

| ResNet patch side | Untargeted ASR | Ten-target macro TASR |
|---|---:|---:|
| 3 | 8.66% | 1.48% |
| 5 | 25.80% | 6.24% |
| 7 | 46.56% | 24.97% |
| 16 | 89.79% | 99.92% |

Values average three patch seeds; targeted and untargeted patches are optimized separately. ASR uses the undefended model's clean-correct set; TASR additionally excludes images whose true class is the target. A 16-by-16 patch has 25% nominal image area. Matched random/mean-color controls, all targets and all seeds remain in the results.

The ensemble extension **does not outperform ResNet-only optimization** on held-out DenseNet side-7 targeted macro averages: 7.87% vs 7.97% at fixed steps, and 7.82% vs 8.48% at matched training time. Defense-aware attacks are not uniformly stronger than EOT-matched controls. Both defenses cost about 2.14 percentage points of clean test accuracy; resize/pad also exceeds the 2 pp validation-selection limit and is explicitly reported as the weakest over-limit fallback. These negative and mixed findings are retained.

## A/B/C/D layout and dependencies

| Part | Implementation | Dependencies |
|---|---|---|
| A | `src/data.py`, `src/models/`, model bundles, metrics, evaluator and artifact audits | Official CIFAR-10 |
| B | `src/patches/`, base four-size/all-target matrix | A models and split |
| C | `src/ensemble.py`, fixed-step/time groups | A models, B base patches and shared optimizer |
| D | `src/defenses.py`, adaptive/EOT-matched/strength groups | A ResNet, B renderer/optimizer and base patches |

This repository tracks source, PDFs, compact tables/figures and audit summaries. Large raw records and trained tensors are distributed through the release. For the archived reproduction, download `A_Code.zip` through `D_Code.zip` and extract them in that order into the same parent directory; they share the `ECE661_Project` root. Merge `CIFAR10_Dataset.zip` and `Shared_Results.zip` from the complete delivery there. See [run order](RUN_ORDER_CN.txt) and [detailed local guide](LOCAL_GPU_CN.md). Those instructions describe the original delivery layout; in this Git checkout, PDFs live under `reports/`.

## Reproduce from source

The measured computation environment was Windows / RTX 5080 16 GB, Python 3.13.5, PyTorch `2.10.0.dev20251010+cu130`, CUDA runtime 13.0. Exact packages are in [requirements-local-tested.txt](requirements-local-tested.txt); document tooling uses a separate environment in [requirements-docs-tested.txt](requirements-docs-tested.txt). Preserve a known working installation and verify CUDA compatibility before changing it.

From a **fresh** project directory without existing frozen artifacts/results:

```powershell
python -m scripts.local_gpu check
python -m scripts.local_gpu prepare
python -m scripts.local_gpu train
python -m scripts.local_gpu calibrate
python -m scripts.local_gpu attack
python -m scripts.local_gpu final
python -m scripts.analyze_results
python -m scripts.export_visual_assets
python -m scripts.publication_figures
```

This checkout includes frozen JSON manifests and completed result summaries for inspection. Do not run fresh preparation/calibration over those identities. For complete retraining, use a separate source-only checkout/directory; for archived reevaluation, restore the release artifacts and data first. Model changes require rerunning affected patch optimizations and evaluation, rather than reusing old ASR.

```powershell
python -m pytest -q tests
```

## Evidence and limitations

- [Measured publication data](results/publication_data.json), [per-seed tables](results/tables/), [figures](results/figures/).
- [Test audit](results/evaluation_audit.json), [full-validation audit](results/validation_audit.json), [time-budget audit](results/time_budget_audit.json).
- [Original delivery verification](docs/verification/FINAL_VERIFICATION.json), [PDF visual QA](docs/delivery/QA_APPROVED.json), [source hashes](artifacts/code_provenance.json).
- [Original implementation plans](docs/implementation_plans/) and [detailed source README](docs/SOURCE_README_CN.md).

Classifier seed 17 is one training run per architecture. Patch seeds 101/102/103 do not represent classifier-training uncertainty. Bootstrap intervals resample paired images within classes, retaining all draws and the three fixed patch seeds. Digital attack failure is not a certified or physical-world robustness guarantee.

Training timings are observed synchronized wall times, not GPU-kernel-active time. The legacy `end_to_end_seconds` field spans optimizer-function entry through artifact assembly, excluding external model/data loading and final serialization. Later D fixed-step runs shared the GPU with an interactive graphics workload; these timings do not isolate defense overhead. C matched-time experiments finished before that workload. See the archived resource observations.

`configs/protocol.json` retains immutable experimental bytes and historical status wording; actual completion is established by training completion records, audited raw conditions and final verification, not that historical text. AI assisted implementation, execution, debugging, analysis and writing. Planned member responsibilities do not attest to unobserved individual effort or attendance.
