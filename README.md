# IrEne: Interpretable Energy Prediction for Transformers

This repository contains associated data and code for our [ACL'21 paper](https://arxiv.org/pdf/2106.01199.pdf).

> **Disclaimer**: This the original code we used in the paper. We've cleaned up the code and standardized our dataset format for extensibility and usability, which can be found in `master` branch.


## Installation

```
conda create -n irene python=3.7 -y && conda activate irene
pip install -r requirements.txt
```

## Reproducing IrEne Results

To reproduce IrEne (full model) results, run:

```bash
# For device-1 (qpc)
python ml_level_crossvalidation.py irene_qpc
python non_ml_level_crossvalidation.py irene_qpc # Run after ml_level is complete

# For device-2 (jpc)
python ml_level_crossvalidation.py irene_jpc
python non_ml_level_crossvalidation.py irene_jpc # Run after ml_level is complete
```

## Reproducing all the Results:

To reproduce (almost) all experiments showin in the paper, run the following to generate all experiment configs, run ml-level and non-ml (module, model) level cross-validation, and geneate results report.

```bash
# Generate and save all experiment configs used in the paper.
python generate_all_experiment_configs.py

# Run ml-level cross-validation.
python ml_level_crossvalidation.py __exec_all__ # Try __print_all__ first to see what cmds it runs

# After ml-level cross-validation finishes, run non-ml-level crossvalidation.
python non_ml_level_crossvalidation.py __exec_all__ # Try __print_all__ first to see what cmds it runs

# Create and save summary table of all evaluation results.
python generate_results_summary.py
```

## Citation

If you find this work useful, please cite it using:

```
@misc{cao2021irene,
   title={IrEne: Interpretable Energy Prediction for Transformers},
   author={Qingqing Cao and Yash Kumar Lal and Harsh Trivedi and Aruna Balasubramanian and Niranjan Balasubramanian},
   year={2021},
   eprint={2106.01199},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}
```
