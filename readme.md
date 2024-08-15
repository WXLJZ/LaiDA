# LaiDA: Linguistics-aware In-context Learning with Data Augmentation for Metaphor Components Identification

---

**Authors:** Hongde Liu, Chenyuan He, Feiyang Meng, Changyong Niu, and Yuxiang Jia📬️
> This paper has been accepted by the NLPCC 2024.


## 📋 Table of Contents
- [Introduction](#anchor-introduction)
- [Code Structure](#anchor-code-structure)
- [Environment and Requirements](#anchor-environment-and-requirements)
- [Running](#anchor-running)

---

<a id="anchor-introduction"></a>
## 📌 Introduction

Metaphor Components Identification (MCI) contributes to enhancing machine understanding of metaphors, thereby advancing downstream natural language processing tasks. In this research, leveraging LLMs, a new framework is proposed, named Linguistics-aware In-context Learning with Data Augmentation (LaiDA). Specifically, utilizing ChatGPT and supervised fine-tuning, a high-quality dataset is tailored. Additionally, LaiDA incorporates a metaphor dataset for pre-training. A graph attention network encoder generates linguistically rich feature representations to retrieve similar examples. Subsequently, LLM is fine-tuned with prompts that integrate linguistically similar examples on the meticulously constructed training set. 

---

<a id="anchor-code-structure"></a>
## 📂 Code Structure
```angular2html
├── 📁 data——original data 
├── 📁 data_process——some scripts for data processing
├── 📁 Fine_tuning_data——the data used for task fine-tuning
├── 📁 preprocess_data——the data used in the preprocess stage
├── 📁 pretrain_data——the data used in the pretrain stage
└── 📁 src——the code of method
    ├── 📁 gnnencoder——GAT encoder
    └── 📁 Icl——In-context learning
```

---

<a id="anchor-environment-and-requirements"></a>
## 🛠 Environment and Requirements
- **Python version:** 3.8 or above
- **GPU** One NVIDIA GeForce RTX 4090 24G
- **Dependencies** Refer to `requirements.txt` for the complete list. It is recommended to install directly with the following command:
```shell
pip install -r requirements.txt
```
---

<a id="anchor-running"></a>
## 🚀 Running
><span style="color:red;">Note</span>: Modify the parameter to your owns before running.

### Step1 Data Preprocessing
```shell
CUDA_VISIBLE_DEVICES=0 bash run_data_preprocess.sh
```
```shell
# Important parameter description
--is_process_data # Indicates that the current stage is data preprocessing
--method # random
--prefix # The location of the fine-tuning data
--inst_prefix # The location of the fine-tuning instruction data
--results_path_prefix # The location where the result file is saved
--dataset_dir # The location of the fine-tuning data
--model_name_or_path # The location of the model
```

### Step2 Data Augmentation Pre-training
```shell
CUDA_VISIBLE_DEVICES=0 bash run_pretrain.sh
CUDA_VISIBLE_DEVICES=0 bash run_export_model.sh
```

```shell
# Important parameter description——run_pretrain.sh
--is_pretrained # Indicates that the current stage is pre-training
--method # random
--selected_k # Number of examples selected
--prefix # The location of the fine-tuning data
--inst_prefix # The location of the fine-tuning instruction data
--dataset_dir # The location of the fine-tuning data
--model_name_or_path # The location of the model
# Important parameter description——run_export_model.sh
--model_name_or_path # The location of the model
--adapter_name_or_path # The location of fine-tuned weights file
--export_dir # The location where the merge model is saved
--export_size # The maximum size allowed for a single model file
```

### Step3 Train GAT Encoder
```shell
CUDA_VISIBLE_DEVICES=0 bash run_gnn.sh
```
```shell
# Important parameter description
--save_path # Location to save the model
--bert_model_path # The location of the bert-base-chinese
--data_path # The location of the train data
```


### Step4 Task Fine-tuning
```shell
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

```shell
# Important parameter description
--method # The in-context examples retrieval methods (gnn, random or bert)
--selected_k # Number of examples selected
--bert_path # The location of the bert-base-chinese
--gnn_path # The location of the GAT encoder (The model saved in step3)
--prefix # The location of the fine-tuning data
--inst_prefix # The location of the fine-tuning instruction data
--results_path_prefix # The location where the result file is saved
--dataset_dir # The location of the fine-tuning data
--model_name_or_path # The location of the model
```



