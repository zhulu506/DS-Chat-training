# [DS-Chat-training](#ds-chat-training)

> 相应文档：[使用 DeepSpeed 微调 OPT 基础语言模型](https://blog.csdn.net/weixin_43254181/article/details/144494212)

- [DS-Chat-training](#ds-chat-training)
  - [第 1 次提交](#第-1-次提交)
  - [第 2 次提交](#第-2-次提交)

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

> 2024-12-18 | 1 | zjma

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