# ErrorLLM - Modeling SQL Errors for Text-to-SQL Refinement
The complete source code for SIGKDD2026 research track accepted paper: "ErrorLLM: Modeling SQL Errors for Text-to-SQL Refinement". 

## Enviourmental Setups

```bash
conda create -n errorllm python=3.10 -y
conda activate errorllm
pip install -r requirements.txt
```

Download the required model weights into the model/ directory:

CodeS-7B (base model for error detection fine-tuning):
```bash
mkdir -p model
huggingface-cli download seeklhy/codes-7b --local-dir model/codes-7b
```

ELECTRA-large (PLM encoder for QSS construction):
```bash
huggingface-cli download google/electra-large-discriminator --local-dir model/electra-large-discriminator
```

Download the [BIRD](https://bird-bench.github.io/) benchmark and place it under the database/ directory:
```
database/
  bird/
    train/
      train.json
      train_tables.json
      train_gold.sql
      train_databases/
    dev/
      dev.json
      dev_tables.json
      dev_gold.sql
      dev_databases/
```
The train split is used for data synthesis and ErrorLLM training; the dev split is used for inference and evaluation.

## LLaMA-Factory Plugin

`src/error_detection/` is a custom training module for [LLaMAFactory](https://github.com/hiyouga/LlamaFactory). After installing LLaMAFactory (please setup the llamafactory enviourment according to their repository):

```bash
pip install llamafactory
```

Copy the folder into the LLaMA-Factory train directory:

```
<llamafactory_path>/src/llamafactory/train/error_detection/
```

## Structure

```
ErrorLLM/
  run.sh                    
  requirements.txt
  src/
    ast_construction.py     
    evaluation.py           
    m_schema/               
    qss_construction/       
    rules/                  
    data_synthesize/        
    refinement/             
    training/               
    error_detection/        
    utils/                  
```

## Usage

```bash
export OPENAI_API_KEY="your-key"
bash run.sh
```
