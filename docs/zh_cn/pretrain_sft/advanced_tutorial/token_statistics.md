# Token 长度统计

通过 `TokenStatsConfig` 记录样本首次截断前的 `original_num_tokens`，按数据集统计长度分布和截断损失。现有 `num_tokens`、训练输入及 packing 行为不变。

尚未采集原始长度时，使用配置模式从原始数据生成统计；已有长度缓存时，可以直接读取缓存生成报告。两种方式都是离线数据处理，不启动模型训练。

## 从原始数据生成统计

此方式加载 JSONL 数据和 tokenizer，通过 `TokenStatsConfig` 采集截断前长度，生成缓存并导出报告；已有匹配的统计缓存时会复用。

在已安装 XTuner 数据依赖的环境中，将下面的代码保存为 `stats_config.py`。这是自行创建的配置文件，按下方命令使用时放在仓库根目录。示例采用 FTDP，数据路径、tokenizer、模板、长度上限和采样参数应与训练配置一致。

```python
from pathlib import Path

from transformers import AutoTokenizer
from xtuner.v1.datasets import DatasetConfig, FTDPTokenizeFnConfig, build_datasets
from xtuner.v1.datasets.token_stats import TokenStatsConfig


def build_stats_inputs(work_dir: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained("<tokenizer 路径>")
    dataset = DatasetConfig(name="subset-a", anno_path="<JSONL 路径>", cache_dir=work_dir / "cache")
    tokenize_fn = TokenStatsConfig(FTDPTokenizeFnConfig(chat_template="qwen", max_length=2048))
    return {"datasets": build_datasets([{"dataset": dataset, "tokenize_fn": tokenize_fn}], tokenizer)}
```

从仓库根目录执行：

```shell
XTUNER_DETERMINISTIC=false XTUNER_TOKENIZE_WORKERS=4 python -m xtuner.tools.token_stats \
  --config stats_config.py --output-dir work_dirs/token_stats
```

新字段保存为 `cache_dir/<file_hash>/<tokenize_hash>/jsonl_meta/original_num_tokens.npy`，使用独立的统计缓存，不覆盖原训练缓存。需要 packing 统计时，配置函数再返回现有的 `packed_dataset` 和 `dataloader_config`。

## 从已有缓存生成统计

此方式直接读取已有 `jsonl_meta` 目录中的长度数组，重新汇总并导出 `subset_token_stats.csv`，不加载 tokenizer，也不重新处理原始数据。

创建 `cache_manifest.json`，同一 subset 的多个 shard 放入同一个 `meta_dirs`：

```json
[
  {"name": "subset-a", "meta_dirs": ["cache/file_hash/tokenize_hash/jsonl_meta"]}
]
```

```shell
python -m xtuner.tools.token_stats \
  --cache-manifest cache_manifest.json --output-dir work_dirs/token_stats --workers 4
```

缓存路径相对于 manifest 所在目录。旧缓存缺少 `original_num_tokens` 时，原始长度和截断损失记为未知；需要补充这些信息时，使用前面的配置模式从原始数据采集。此模式不生成 packing 报告，packing 统计需要配置模式提供实际的打包数据集。CLI 单进程启动，不使用 `torchrun`。

## 统计结果

结果写入输出目录下的 `token_stats_<时间>_<随机后缀>/`，日志打印覆盖情况和文件路径：

| 文件 | 内容 | 统计范围 |
| --- | --- | --- |
| `subset_token_stats.csv` | 每个 subset 一行：样本数、已知/未知数量、原始 token 总数、均值、中位数、P99、最大值，以及截断数量和比例。 | 配置模式统计过滤/采样后的实例；缓存模式统计过滤/采样前的原始记录。 |
| `packing_token_stats.csv` | 有效长度分布，以及 packing/collator 的部分裁切、整条丢弃、label shift 和 padding。 | 仅提供 packing 配置时生成，每个 pack 统计一次，位于分布式 sampler 之前。 |

仅对原始长度已知的样本计算以下指标（缓存值 `-1` 也表示未知）：

```text
单条截断损失 = original_num_tokens - num_tokens
截断样本比例 = 损失大于 0 的样本数 / 已知原始长度的样本数
截断 token 比例 = 截断损失之和 / 对应样本的原始 token 数之和
```

原始长度恰好等于上限时没有截断。没有已知样本时，相关指标留空。Packing 有效长度排除 padding；后续裁切单独统计，不用原始长度减最终 pack 长度，也不把整条丢弃或 label shift 算作截断。

## 实现说明

以下路径相对于仓库根目录：

- `xtuner/v1/datasets/token_stats.py`：采集原始长度。
- `xtuner/v1/datasets/data_item.py`、`xtuner/v1/datasets/jsonl.py`：字段声明与缓存。
- `xtuner/tools/token_stats.py`：汇总和导出。

支持纯文本 OpenAI、FTDP 和标准预训练；packing 支持 `none/soft/hard/__legacy` 的文本 collator。自定义 tokenization、多模态、preset packing 和 LongText chunk 暂不支持。统计复用一次完整 tokenize；FTDP 会为完整序列构建 IDs/labels，增加瞬时内存开销。精确分位数需要合并同一 subset 的长度数组，packing 统计还会执行一次数据遍历。
