# [DS-Chat-training](#ds-chat-training)

> 相应文档：[使用 DeepSpeed 微调 OPT 基础语言模型](https://blog.csdn.net/weixin_43254181/article/details/144494212)

- [DS-Chat-training](#ds-chat-training)
  - [第 1 次提交](#第-1-次提交)
  - [第 2 次提交](#第-2-次提交)
  - [第 3 次提交](#第-3-次提交)
  - [第 4 次提交](#第-4-次提交)

## 第 1 次提交

> 2024-12-17 | 1 | zjma

根据我们的实验需要，修改了 [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) 的文件结构和部分代码。

当前的目录结构为：

```
├── dschat
│   └── utils
├── README.md
└── step1_supervised_finetuning
    ├── Dahoas      # dataset
    ├── facebook    # pre-trained model
    ├── main.py
    ├── training_scripts
    └── yitingxie   # dataset
```

运行脚本方式：

1、下载该仓库。

```
git clone https://github.com/zhulu506/DS-Chat-training.git
```

2、准备预训练模型和数据集。

3、运行脚本。

```
pwd # DS-Chat-training/step1_supervised_finetuning
bash training_scripts/my_test/run_opt-125m.sh
```

## 第 2 次提交

> 2024-12-25 | 1 | zjma

1、修改了`step1_supervised_finetuning/main.py`中的`parse_args()`函数，增加了`--gradient_checkpointing`和`--only_optimize_lora`不能同时存在的判断，增强鲁棒性。

2、修改了`step1_supervised_finetuning/main.py`中的训练流程：
- 每个 step 输出`max_memory_reserved`和`max_memory_allocated`。
- 每 10 个 step 输出`memory_summary`。
- 每 100 个 step 输出总的训练时间和吞吐量，并结束训练。

3、修改了 GMLake 的微调训练脚本`finetune.sh`，支持开启`GMLAKE`和不开启`GMLAKE`两种模式，支持`opt-125m`、`opt-350m`、`opt-1.3b`三种模型。

4、新 4 个测试脚本：
- `training_scripts/torch_ori.sh`用于测试 PyTorch；
- `training_scripts/torch_es.sh`用于测试 PyTorch + es；
- `training_scripts/gmlake_ori.sh`用于测试 GMLake；
- `training_scripts/gmlake_pro.sh`用于测试 Our Approach；

## 第 3 次提交

小修改，在每次运行脚本前先执行清除环境变量`unset PYTORCH_CUDA_ALLOC_CONF`。

## 第 4 次提交

> 2024-12-26 | 1 | zjma

1、新增目前支持的模型：

| Model | Parameters  | HF-Mirror |
|--|--|--|
| OPT-125M | 125M | [facebook/opt-125m](https://hf-mirror.com/facebook/opt-125m) |
| OPT-350M | 350M | [facebook/opt-350m](https://hf-mirror.com/facebook/opt-350m) |
| OPT-1.3B | 1.3B | [facebook/opt-1.3b](https://hf-mirror.com/facebook/opt-1.3b) |
| GPT-Neo 125M | 125M | [EleutherAI/gpt-neo-125m](https://hf-mirror.com/EleutherAI/gpt-neo-125m) |
| GPT-Neo 1.3B | 1.3B | [EleutherAI/gpt-neo-1.3B](https://hf-mirror.com/EleutherAI/gpt-neo-1.3B) |
| GPT-Neo 2.7B | 2.7B | [EleutherAI/gpt-neo-2.7B](https://hf-mirror.com/EleutherAI/gpt-neo-2.7B) |
| GPT-2 | 124M | [openai-community/gpt2](https://hf-mirror.com/openai-community/gpt2) |
| GPT-2 Medium | 355M | [openai-community/gpt2-medium](https://hf-mirror.com/openai-community/gpt2-medium) |
| GPT-2 Large | 774M | [openai-community/gpt2-large](https://hf-mirror.com/openai-community/gpt2-large) |
| GPT-2 XL | 1.5B | [openai-community/gpt2-xl](https://hf-mirror.com/openai-community/gpt2-xl) |

2、新增 EleutherAI/gpt-neo 训练脚本`/training_scripts/my_test/run_gpt-neo.sh`，修改`main.py`中的部分代码解决报错：

```python
# 复现 EleutherAI/gpt-neo-125m 时遇到一个报错 AttributeError: 'DeepSpeedEngine' object has no attribute 'model'
# 将 model.model 替换为 model.module 可以解决这个问题。
# print_throughput(model.model, args, end - start,
#                  args.global_rank)
print_throughput(model.module, args, end - start,
                  args.global_rank)
```

3、新增 openai-community/gpt2 训练脚本`/training_scripts/my_test/run_gpt2.sh`。复现`/openai-community/gpt2-medium`时遇到报错：`UnboundLocalError: local variable 'tokenizer' referenced before assignment`，修改`dschat/utils/utils.py`中的部分代码解决报错：

```python
# def load_hf_tokenizer(model_name_or_path,
#                       fast_tokenizer=True,
#                       add_special_tokens=None):
#     if os.path.exists(model_name_or_path):
#         # Locally tokenizer loading has some issue, so we need to force download
#         model_json = os.path.join(model_name_or_path, "config.json")
#         if os.path.exists(model_json):
#             model_json_file = json.load(open(model_json))
#             model_name = model_json_file.get("_name_or_path",
#                                              model_name_or_path)
#             tokenizer = get_tokenizer(model_name,
#                                       fast_tokenizer=fast_tokenizer)
#     else:
#         tokenizer = get_tokenizer(model_name_or_path,
#                                   fast_tokenizer=fast_tokenizer)

#     if add_special_tokens is not None:
#         add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
#             else add_special_tokens
#         tokenizer.add_special_tokens(
#             {'additional_special_tokens': add_special_tokens})

#     return tokenizer

def load_hf_tokenizer(model_name_or_path,
                      fast_tokenizer=True,
                      add_special_tokens=None):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file.get("_name_or_path",
                                             model_name_or_path)
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=fast_tokenizer)
        else:
            raise FileNotFoundError(
                f"Configuration file 'config.json' not found in {model_name_or_path}.")
    else:
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})

    return tokenizer
```

4、复现`/facebook/opt-350m`时遇到报错：`OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like opt-350m is not the path to a directory containing a file named config.json.`：

修改`/facebook/opt-350m/config.json`中的`"_name_or_path": "opt-350m"`为`"_name_or_path": "facebook/opt-350m"`。