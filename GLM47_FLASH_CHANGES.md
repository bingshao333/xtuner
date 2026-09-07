# GLM-4.7-Flash 适配改动报告

## 概要

相较于官方 XTuner 基线，当前工作树新增了 GLM-4.7-Flash 的原生模型配置与 SFT 对话适配，并扩展 MLA 的 RMSNorm epsilon 配置。改动共涉及 6 个已修改文件和 2 个新增文件，当前均尚未形成 Git commit。

## 改动内容

### 1. 原生模型支持

- 新增 `xtuner/v1/model/moe/glm47_flash.py`。
- 基于 DeepSeekV3 的 MLA/MoE 实现定义 `Glm47Flash` 与 `Glm47FlashConfig`。
- 对接 Hugging Face `glm4_moe_lite` 配置，可读取和导出模型配置。
- 覆盖 GLM-4.7-Flash 的词表、层数、MLA、MoE、RoPE、路由器及 MTP 参数。
- 增加 MTP 权重键到 Hugging Face 格式的映射。
- 在 `xtuner/v1/model/__init__.py` 注册模型名 `glm-4.7-flash`，并支持从 `model_type=glm4_moe_lite` 自动识别。

### 2. MLA 数值配置

- 在 `xtuner/v1/module/attention/mla.py` 为 `MLAConfig` 和 `MultiLatentAttention` 增加 `rms_norm_eps`，默认值为 `1e-6`。
- MLA 中的 Q/KV 压缩分支 RMSNorm 改为使用该 epsilon。
- GLM-4.7-Flash 配置显式使用 `1e-5`，与其 Hugging Face 配置保持一致。

### 3. GLM-4.7 对话模板与 SFT

- 新增 `xtuner/v1/data_proto/messages/glm47_chat.py`。
- 实现 GLM-4.7 的 system/user/assistant/tool 消息渲染、工具调用、工具结果、思考内容和 SFT loss mask。
- 在 `xtuner/v1/data_proto/messages/__init__.py` 导出 `Glm47ChatMessages`。
- 在 `xtuner/v1/data_proto/templates/__init__.py` 注册 `glm4.7` 模板，使用 `<|endoftext|>` 和 `<|observation|>` 作为停止词。
- 在 `xtuner/v1/datasets/sft_tokenize_fn/openai.py` 增加 `glm4.7` tokenizer 分支。
- 在 `xtuner/v1/train/arguments/arguments.py` 将 `glm4.7` 加入可选 `chat_template`。

## 脚本代码修改

脚本位于 `/mnt/shared-storage-user/shaobing-p/scripts/`，主要包括：

- `glm47_flash_grpo_gsm8k_long.py`：基于 GSM8K 配置构造 GLM-4.7-Flash 的 GRPO 长任务，支持通过环境变量调整步数、批量、采样长度、学习率和评估间隔；关闭 MTP、编译及额外 loss，并保留最终 HF 导出。
- `glm47_flash_sft_gsm8k_formal.py`：定义 GLM-4.7-Flash 全参数 SFT，使用 `glm4.7` 模板，关闭 MTP，允许忽略官方 checkpoint 中的 MTP 权重，并仅保留最终 HF 导出。
- `run_glm47_rl_long.sh`、`run_glm47_sft_formal.sh`：完成模型/数据校验、环境变量配置、多卡启动、Ray 或 torchrun 调度，以及 W&B 与 JSONL 日志记录。
- `submit_glm47_rl_long_rjob.sh`、`submit_glm47_sft_formal_rjob.sh`：封装 GPU、CPU、内存、镜像、数据挂载和训练参数，提交集群 RJob。

## 依赖关系与用途

- 历史 RL 实验至少依赖模型注册和 MLA RMSNorm epsilon 改动。
- GLM-4.7-Flash 的 SFT 训练额外依赖对话模板注册、消息渲染及 tokenizer 分支。
- 若缺少新增模型文件，无法完成 `glm-4.7-flash` 原生配置构建或 Hugging Face 配置转换。

## 当前状态

- 已修改文件：6 个。
- 新增未跟踪文件：2 个。
- Git commit：尚未创建。
