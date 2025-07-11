# Official Code for "Agent-Centric Actor-Critic (ACAC) for Asynchronous Multi-Agent Reinforcement Learning" (ICML 2025)

This repository contains the official PyTorch implementation for the paper "Agent-Centric Actor-Critic (ACAC) for Asynchronous Multi-Agent Reinforcement Learning," accepted at the International Conference on Machine Learning (ICML) 2025.

The paper can be found at [here](https://openreview.net/forum?id=323GZNnGqe)

## ⚙️ Installation
To get started, create a Conda environment and install the necessary dependencies.

```bash
# 1. Create and activate a new conda environment
conda create -n acac python=3.9
conda activate acac

# 2. Install the package in editable mode
pip install -e .

# 3. Install PyTorch and related libraries
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 numpy=1.21.5 -c pytorch -c nvidia
```


## ▶️ How to Run
You can run experiments by specifying the algorithm, environment, and other settings via command-line arguments.

### General Command Structure:
```bash
python scripts/acac_main.py --exp_name=<your_experiment_name> --alg_name=<algorithm> --env_name=<environment> --seed=<seed_number>
```

### Example:
To run our proposed ACAC algorithm on the Overcooked environment (7x7, Map A) with seed 0, use the following command:
```bash
python scripts/acac_main.py --exp_name='acac_on_overcooked_A' --alg_name='acac' --env_name='ovcA7' --seed=0
```

## 🛠️ Available Arguments
Below is a list of available arguments you can use to configure the experiments.

### Logging Arguments
- `--exp_name`: Sets the name for the experiment, which is used for creating logging directories.
- `--wandb`: (Optional) Enables logging with Weights & Biases.
- `--wandb_project`: (Optional) Sets the wandb project name (e.g., `--wandb_project='acac_marl'`).


### Algorithm Arguments
Use the `--alg_name` flag to select the desired algorithm.

| Algorithm | Argument |
|---|---|
| ACAC (Ours) |	`--alg_name='acac'` |
| ACAC (Vanilla) | `--alg_name='acac_vanilla'` |
| ACAC (Micro-level GAE) | `--alg_name='acac_micro_gae'` |
| ACAC (Duplicate) | `--alg_name='acac' --duplicate` |

### Environment Arguments
Use the `--env_name` flag to select the environment.

📦 BoxPushing
| Environment | Size | Argument |
| :--- | :-: | :--- |
| BoxPushing | 6x6 | `--env_name='bp6'` |
| BoxPushing | 8x8 | `--env_name='bp8'` |
| BoxPushing | 10x10 | `--env_name='bp10'` |

🍳 Overcooked
| Environment | Map | Argument |
| :--- | :-: | :--- |
| Overcooked | Map A | `--env_name='ovcA7'` |
| Overcooked | Map B | `--env_name='ovcB7'` |
| Overcooked | Map C | `--env_name='ovcC7'` |
| Overcooked-Rand | Map A | `--env_name='ovcAR7'` |
| Overcooked-Rand | Map B | `--env_name='ovcBR7'` |
| Overcooked-Rand | Map C | `--env_name='ovcCR7'` |
| Overcooked-Large | Map A | `--env_name='ovcA11_N6'` |
| Overcooked-Large | Map B | `--env_name='ovcB11_N6'` |
| Overcooked-Large | Map C | `--env_name='ovcC11_N6'` |
| Overcooked-Large-Rand | Map A | `--env_name='ovcAR11_N6'` |
| Overcooked-Large-Rand | Map B | `--env_name='ovcBR11_N6'` |
| Overcooked-Large-Rand | Map C | `--env_name='ovcCR11_N6'` |

## 🙏 Acknowledgement

Our implementation is built by modifying and extending several outstanding open-source projects to fit our research goals. We are deeply grateful to the developers of the following repositories, which provided a crucial foundation for our work.

| Repository	| How It Was Used |
| :--- | :--- |
| [MacroMARL](https://github.com/yuchen-x/MacroMARL) | Adapted as foundational code for our algorithms. |
| [gym-macro-overcooked](https://github.com/WeihaoTan/gym-macro-overcooked) | Used as a base for our custom MARL environment. |
| [rlkit](https://github.com/rail-berkeley/rlkit) & [rllab](https://github.com/rll/rllab) | Modified and used for experiment logging. |


## ✍️ Citation
If you use this codebase, please cite our paper:
```text
@inproceedings{junghongyoon2025acac,
  title={Agent-Centric Actor-Critic for Asynchronous Multi-Agent Reinforcement Learning},
  author={Jung, Whiyoung and Hong, Sunghoon and Yoon, Deunsol and Lee, Kanghoon and Lim, Woohyung},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## 📧 Contact
If you have any questions, please contact us at the email address below:

{whiyoung.jung, sunghoon.hong, dsyoon, kanghoon.lee}@lgresearch.ai

