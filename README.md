# Memorize-then-Generalize Framework

This repository contains the codebase for the paper **"Rote Learning Considered Useful: Generalizing over Memorized Knowledge in LLMs"**, which explores how large language models (LLMs) can first memorize factual knowledge and then generalize from it. The overall pipeline is divided into three main components:

1. **Data Generation** – for creating synthetic or real-world factual knowledge data.
2. **Training** – for fine-tuning the LLM in two stages: memorization followed by generalization.
3. **Evaluation** – for measuring memorization accuracy and generalization ability.

To implement the **memorize-then-generalize** training paradigm, modify the configuration files located in the `training/` directory. These configs control which dataset is used at each stage and how the model is trained, enabling a two-stage fine-tuning process where the model first learns to memorize a set of facts and then is trained to generalize beyond them.

---

## 🧩 Repository Structure

```
open_memorize-then-generalize/
├── data_generation/
│   ├── get_description/
│   ├── generate_wiki.py, generate.py, random_object.py
│   ├── syn_data_main.py
├── training/
│   ├── main/
│   ├── config_generation.py
│   ├── pipeline_train_custome.py
│   ├── write_path.py
├── evaluation/
│   ├── eval.py, eval_gen.py, eval_all_loss.py
│   ├── test_examples/, test_result/, conf/
├── util_public/
├── models.json
```

---

## 📦 Setup

### 1. Create Conda Environment

```bash
conda create -n memorize-gen python=3.10 -y
conda activate memorize-gen
pip install -r requirements.txt
```

> ⚠️ Note: You may need to manually install dependencies for your LLM backend (e.g., `transformers`, `vllm`, or `accelerate`).

---

## 🔧 1. Data Generation

Synthetic or semi-synthetic knowledge data is generated for training and testing.

```bash
cd data_generation
bash gen_script.sh
```

Key files:
- `generate.py`, `generate_wiki.py`: for generating source-target fact pairs
- `random_object.py`: auxiliary alternative objects list generator
- `get_description/`: functions for pulling descriptions of entities
- Output is saved into `.csv` files for training and testing.
- Change the configuration in `._gen_config.yaml` file.

---

## 🏋️ 2. Training

This step provide different training approach for a base LLM to memorize factual data.
We provide the script which used for submit the job on a slurm system.


```bash
cd training/main
sbatsh ../../pipeline_llama_custome.sh
```

Key files:
- `config_generation.py`: builds model and training config
- `construct_dataset_custome.py`: formats dataset
- `pipeline_train_custome.py`: trains the model
- `write_path.py`: sets paths for outputs

Key configurations:
- `loss_computation`: `normal`, compute all token's loss in the training sequence, as same as the traditional unsupurvised training loss; `s-o`, only compute the loss on the subject and object tokens; `o`, only compute the loss on the object tokens, which is similar to the supervised fine-tuning. 
- `train_text_type`: change the training prompt to what you want

---

## 📈 3. Evaluation

Evaluate model generalization, memorization accuracy, and latent knowledge extraction.

```bash
cd evaluation
python eval.py
```

Optional scripts:
- `eval_all_loss.py`: analyze training loss per example
- `eval_gen.py`: evaluate model generalization capability
- `latent_knowledge_extractor/`: inspect internal knowledge representations

Results are saved in the `test_result/` directory.


---

## 📁 Configuration

All configurations (model name, tokenizer path, batch size, etc.) are stored in:
- `training/main/conf/`
- `evaluation/conf/`
- `models.json`: maps model names to paths and hyperparameters

---

## Dataset

You can download the dataset used in the paper in: https://huggingface.co/datasets/QinyuanWu/T-Rex-Fiction

Download it to your dataset path, and change the path of the dataset path in all the conf files. 
