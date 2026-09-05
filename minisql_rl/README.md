# MiniSQL-RL

MiniSQL-RL 是一个基于 MiniMind 的轻量级 Text-to-SQL Agent 项目。当前阶段提供可复现的电商 SQLite 数据库、只读 SQL 沙箱、面向模型提示词的 Schema 导出，以及一组执行结果确定的种子评测题。

## 数据库包含什么

数据库覆盖 10 张业务表和 2 个统计视图：

| 对象 | 业务含义 |
|---|---|
| `users` | 用户、城市和会员等级 |
| `categories` | 两级商品分类 |
| `products` | 商品、品牌、成本价和标价 |
| `warehouses` | 仓库信息 |
| `inventory` | 分仓库存和安全库存 |
| `orders` | 订单状态、时间和收货城市 |
| `order_items` | 商品数量、成交价和明细优惠 |
| `payments` | 实付金额及支付方式 |
| `refunds` | 退款原因、金额和审核状态 |
| `reviews` | 商品评分和评价 |
| `order_amounts` | 订单应付金额视图 |
| `product_sales` | 商品累计销售视图 |

所有模拟数据由固定随机种子生成。相同参数会得到相同的业务数据，便于对 SFT、Schema Linker 和 GRPO 实验做公平比较。

## 快速开始

在仓库根目录运行：

```bash
python -m minisql_rl.database.build
```

默认生成：

```text
minisql_rl/data/generated/ecommerce.db
minisql_rl/data/generated/benchmark_seed.jsonl
minisql_rl/data/generated/schema_context.txt
```

重新生成已有数据库时必须显式传入 `--overwrite`：

```bash
python -m minisql_rl.database.build --overwrite
```

可调整数据规模和日期范围：

```bash
python -m minisql_rl.database.build \
  --users 1000 \
  --products 120 \
  --orders 10000 \
  --start-date 2025-01-01 \
  --end-date 2026-08-31 \
  --seed 20260905 \
  --overwrite
```

## 安全执行模型生成的 SQL

```python
from minisql_rl.database import SQLSandbox

sandbox = SQLSandbox("minisql_rl/data/generated/ecommerce.db")
result = sandbox.execute(
    "SELECT shipping_city, COUNT(*) AS n "
    "FROM orders GROUP BY shipping_city ORDER BY n DESC"
)
print(result.to_dict())
```

沙箱只接受 `SELECT` 和 `WITH`，并同时使用以下保护：

- SQLite 只读连接；
- SQLite authorizer 拦截写操作、DDL、`ATTACH` 和 `PRAGMA`；
- 单次查询超时；
- SQL 长度和返回行数限制；
- 拒绝多语句执行。

这层保护适合训练和本地演示，但不是面向不可信公网流量的完整数据库安全边界。

## 运行测试

```bash
python -m unittest discover -s minisql_rl/tests -v
```

测试覆盖建库完整性、外键、可复现性、Schema 导出、危险 SQL 拦截、结果截断，以及所有种子评测 SQL 的真实执行。

## 生成训练数据

先完成数据库构建，再运行参数化训练数据流水线：

```bash
python -m minisql_rl.database.build --overwrite
python -m minisql_rl.data_pipeline.build
```

默认生成 1,500 条经过数据库真实执行的数据：

- `train`：1,200 条，覆盖 11 个常见查询模板族；
- `dev`：150 条，覆盖 3 个训练集未出现的组合查询模板族；
- `test`：150 条，覆盖 3 个更复杂的隐藏模板族；
- SFT 训练集中约 25% 为“错误 SQL → 数据库报错 → 修正 SQL”的多轮轨迹。

生成目录中的主要文件：

| 文件 | 用途 |
|---|---|
| `canonical_train/dev/test.jsonl` | 问题、标准 SQL、执行结果和结果哈希，是评测真值源 |
| `sft_train.jsonl` | MiniMind `SFTDataset` 可读取的 Tool Call 和纠错训练数据 |
| `sft_dev.jsonl` | 开发集 SFT 格式数据，用于检查 loss 或人工抽样 |
| `agent_rl_train.jsonl` | SQL Agent-RL Prompt、参考 SQL 和结果哈希 |
| `eval_dev_prompts.jsonl` | 不包含标准 SQL 的开发集推理输入 |
| `eval_test_prompts.jsonl` | 不包含标准 SQL 的测试集推理输入 |
| `manifest.json` | 配置、数据分布、拒绝原因、文件哈希和泄漏检查 |

可自定义规模：

```bash
python -m minisql_rl.data_pipeline.build \
  --train-size 8000 \
  --dev-size 500 \
  --test-size 500 \
  --repair-ratio 0.30
```

流水线不会直接信任模板生成的 SQL。每个样本都要先通过只读沙箱真实执行；执行失败、空结果、无信息标量、结果过大和重复样本会被拒绝并重新采样。

数据集按 SQL 模板族切分，而不是先生成全部样本再随机切行，因此同一模板族不会同时出现在训练集和测试集。这会让测试分数更接近组合泛化能力，避免只替换日期和数字造成的数据泄漏。

独立复验全部标准 SQL 和数据格式：

```bash
python -m minisql_rl.data_pipeline.validate
```

`agent_rl_train.jsonl` 已经保存 SQL 执行奖励所需的 `reference_sql` 和 `expected_result_hash`，但不能直接交给原版 `trainer/train_agent.py`：原脚本只会执行天气、计算器等模拟工具。后续需要接入 `query_database` 工具和基于结果哈希的 SQL Reward。

## 数据口径

- 明细销售额：`quantity * unit_price - discount_amount`；
- 有效订单：默认指 `status != 'cancelled'`；
- 实付金额：明细销售额之和减订单优惠，再加运费；
- 退款统计应明确是否只计算 `refunds.status = 'approved'`；
- 日期字段使用 `YYYY-MM-DD HH:MM:SS` 文本格式，可使用 SQLite 的 `strftime`。

## 开源说明

本项目基于 Apache-2.0 协议的 [MiniMind](https://github.com/jingyaogong/minimind) 进行扩展，保留原仓库许可证和归属说明。MiniSQL-RL 新增的电商数据层、SQL 沙箱、评测集及后续训练代码会在提交记录和文档中明确标注。
