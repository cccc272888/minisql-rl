# MiniSQL-RL

MiniSQL-RL 是一个基于 MiniMind 的轻量级 Text-to-SQL 项目。模型接收完整的电商数据库 Schema 和自然语言问题，直接生成只读 SQLite 查询；外部执行器负责安全运行 SQL，并以执行结果一致性作为评测指标和后续 GRPO 的可验证奖励。

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

所有模拟数据由固定随机种子生成。相同参数会得到相同的业务数据，便于对通用 SFT、SQL SFT 和 GRPO 实验做公平比较。

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

默认生成 2,250 条经过数据库真实执行的数据：

- `train`：1,200 条，覆盖 21 个 easy / medium / hard 可训练查询族；
- `dev`：150 条，覆盖相同的 21 个查询族，但问题与 SQL 参数组合不和训练集重复；
- `challenge`：150 条，复现第一版的 3 个困难查询族，用于分析和课程训练调参；
- `composition_train`：600 条，与 Challenge 同查询族但参数组合完全不重复；
- `test`：150 条，覆盖另外 3 个只重组已见基础算子的隐藏查询族；
- SFT 输入始终提供完整数据库 Schema，避免提前泄漏答案涉及的表；输出只包含标准 SQL。

生成目录中的主要文件：

| 文件 | 用途 |
|---|---|
| `canonical_train/dev/challenge/composition_train/test.jsonl` | 问题、标准 SQL、执行结果、算子标签和结果哈希 |
| `sql_sft_train.jsonl` | MiniMind `SFTDataset` 可读取的直接 SQL 监督数据 |
| `sql_sft_dev.jsonl` | 开发集直接 SQL 监督数据，用于检查 loss 或人工抽样 |
| `sql_composition_sft_train.jsonl` | 三个已知困难组合族的非重叠训练样本 |
| `sql_curriculum_stage2_sft_train.jsonl` | 基础8,000条与组合样本的确定性混合回放数据 |
| `sql_rl_train.jsonl` | GRPO 输入、参考 SQL、样本信息和预期结果哈希 |
| `eval_dev_prompts.jsonl` | 不包含标准 SQL 的开发集推理输入 |
| `eval_challenge_prompts.jsonl` | 第一版困难查询族的推理输入，用于调参和错误分析 |
| `eval_test_prompts.jsonl` | 不包含标准 SQL 的测试集推理输入 |
| `manifest.json` | 配置、数据分布、SQL 算子覆盖、文件哈希和泄漏检查 |

可自定义规模：

```bash
python -m minisql_rl.data_pipeline.build \
  --train-size 8000 \
  --dev-size 500 \
  --challenge-size 500 \
  --composition-train-size 2400 \
  --test-size 500
```

流水线不会直接信任模板生成的 SQL。每个样本都要先通过只读沙箱真实执行；执行失败、空结果、无信息标量、结果过大和重复样本会被拒绝并重新采样。

训练集和开发集覆盖相同的21个查询族，用于训练和超参数选择，但通过全局去重保证两者没有重复的
“问题 + SQL”参数组合。新增的桥接查询族分别教授 `COUNT(DISTINCT)`、`LEFT JOIN`、条件聚合、
聚合比率、`HAVING`、月份分桶和用户级聚合。Challenge 保留第一版测试中的三个困难组合，用于衡量
课程 SFT 是否解决已知失败；新的 Test 使用三个完全不同的组合族。

生成器为每个查询族声明 SQL 基础算子集合，并强制检查 Challenge/Test 的所有基础算子都已在
Train 出现，同时保持查询族完全隔离。这样测试衡量的是“已见算子的未见组合”，而不是要求小模型
零样本发明训练中从未出现的 SQL 语法。

`composition_train` 是第二阶段课程数据：它和 Challenge 共享三个组合查询族，但生成器先保留
Challenge，再生成训练样本，并对所有问题/SQL 参数组合全局去重。因此 Challenge 可以衡量同组合族
下的新参数与新日期泛化；新的 Test 仍与全部训练查询族隔离。

独立复验全部标准 SQL 和数据格式：

```bash
python -m minisql_rl.data_pipeline.validate
```

`sql_rl_train.jsonl` 保存了执行奖励所需的 `expected_result_hash`。SQL 专用训练入口不会把参考 SQL
交给策略模型，也不会使用通用文本 Reward Model；它只把采样 SQL 放入只读沙箱，并根据执行状态与
结果一致性计算可验证奖励。

## 执行结果评测

在领域 SFT 前先评测通用 `full_sft` 权重，保存零领域训练基线。建议先运行 20 条冒烟测试：

```bash
python -m minisql_rl.evaluation.evaluate_model \
  --weight-path out/full_sft_768.pth \
  --split dev \
  --limit 20 \
  --batch-size 4 \
  --output-path logs/sql_baseline_dev_smoke.jsonl
```

冒烟测试通过后移除 `--limit`，运行完整开发集。评测器会用贪心解码批量生成 SQL，将其放入只读沙箱执行，并输出：

- `strict_format_rate`：是否严格只输出 `SELECT/WITH` 查询；
- `sql_extraction_rate`：是否可以从回答中提取查询；
- `executable_rate`：SQL 是否能在目标数据库成功执行；
- `execution_accuracy`：生成 SQL 与标准 SQL 的执行结果值是否一致，列别名差异不影响结果；
- `by_family`：各隐藏查询族的执行正确率。

常规超参数使用 `dev`，组合课程效果使用 `challenge`；最终模型确定后再在新 `test` 上做一次最终
评测，避免根据测试集反复调整训练配置。

第一版领域 SFT 暴露出明显的组合泛化差距：同族 Dev Execution Accuracy 为 98.6%，但同时隐藏
查询族和关键基础算子的旧 Test 为 0%。错误样例表现为日期与 `LIMIT` 参数提取正确，却将整条查询
路由到最相近的商品销售、品牌销售或高价值订单模板。该结果保留为 Challenge 基线，不覆盖、不删除。

## 组合课程 SFT

重新生成 8,000 / 500 / 500 / 500 数据后，从已有领域 SFT 权重继续训练一轮：

```bash
python -m minisql_rl.data_pipeline.build \
  --train-size 8000 \
  --dev-size 500 \
  --challenge-size 500 \
  --composition-train-size 2400 \
  --test-size 500

python -m minisql_rl.data_pipeline.validate

cd trainer
python train_full_sft.py \
  --data_path ../minisql_rl/data/generated/training/sql_sft_train.jsonl \
  --from_weight sql_sft \
  --save_weight sql_curriculum_sft \
  --epochs 1 \
  --batch_size 4 \
  --accumulation_steps 4 \
  --max_seq_len 1536 \
  --learning_rate 1e-6 \
  --num_workers 8 \
  --dtype bfloat16 \
  --save_interval 500 \
  --log_interval 20
```

训练完成后先比较 `dev` 和 `challenge`：`dev` 用于检查旧能力是否遗忘，`challenge` 用于观察三个
已知困难组合是否改善。在确定课程方案前不查看新 `test`。

第一阶段桥接课程完成后，模型在旧版 Dev 保持 98.8%，在21族 Dev 达到94.2%，但 Challenge 仍为
0%。逐例分析显示模型已能生成每个子查询，却只会选择其中一个：用户题漏掉订单门槛，退款率题漏掉
退款聚合，月趋势题漏掉销售额。因此第二阶段使用2,400条完整组合样本，同时回放8,000条基础数据：

```bash
cd trainer
python train_full_sft.py \
  --data_path ../minisql_rl/data/generated/training/sql_curriculum_stage2_sft_train.jsonl \
  --from_weight sql_curriculum_sft \
  --save_weight sql_composition_sft \
  --epochs 1 \
  --batch_size 4 \
  --accumulation_steps 4 \
  --max_seq_len 1536 \
  --learning_rate 5e-7 \
  --num_workers 8 \
  --dtype bfloat16 \
  --save_interval 500 \
  --log_interval 20
```

该阶段不直接训练 Challenge 的500条样本；混合回放用于降低只学三个组合族而遗忘原有21族的风险。

## SQL 执行反馈 GRPO

组合课程 SFT 建立基本泛化能力后，再审计随机采样是否能产生足够的组内奖励差异：

```bash
python trainer/train_sql_grpo.py \
  --audit-only \
  --weight-path out/sql_curriculum_sft_768.pth \
  --max-samples 50 \
  --max-steps 50 \
  --num-generations 4 \
  --temperature 0.8
```

审计结果中的 `active_group_rate` 表示：同一个问题的多条候选 SQL 中，至少出现两档不同奖励的
问题比例。只有这些组能产生非零的组内相对优势。若该值过低，应先提高采样温度、增加候选数，或将
训练样本集中到较难查询族，不能仅凭训练程序在运行就认为 GRPO 正在有效学习。

建议先运行 50 步训练冒烟实验：

```bash
python trainer/train_sql_grpo.py \
  --weight-path out/sql_sft_768.pth \
  --output-path out/sql_grpo_smoke_768.pth \
  --resume-path checkpoints/sql_grpo_smoke_768_resume.pth \
  --max-samples 200 \
  --max-steps 50 \
  --batch-size 1 \
  --num-generations 4 \
  --temperature 0.8 \
  --learning-rate 3e-7
```

奖励分为四级：无法提取 SQL、SQL 执行失败、可执行但结果错误、执行结果正确；严格遵循 SQL-only
输出格式仅获得一个较小的附加奖励。训练只使用 `train`，仍用 `dev` 选择训练步数和采样参数。
最终确定配置后，分别对 SQL SFT 与 GRPO 权重各运行一次隐藏 `test`，报告 GRPO 前后的
Execution Accuracy，不能把 `dev` 样本用于强化学习。

当前已完成的开发集实验结果：

| 权重 | Strict Format | Executable | Execution Accuracy |
|---|---:|---:|---:|
| 通用 SFT（领域训练前） | 0.0% | 0.0% | 0.0% |
| 500 条领域 SFT 冒烟实验 | 100.0% | 100.0% | 83.2% |
| 8,000 条领域 SFT | 100.0% | 100.0% | 98.6% |

其中 98.6% 表示 500 条 in-domain Dev 中有 493 条执行结果正确；另外 7 条均可执行，但查询语义或
结果与标准答案不一致。该权重在旧版三个完全隔离查询族上的 Execution Accuracy 为 0%。空的
`top_errors` 只表示没有 SQL 语法/执行异常，不表示不存在语义错误。

## 数据口径

- 明细销售额：`quantity * unit_price - discount_amount`；
- 有效订单：默认指 `status != 'cancelled'`；
- 实付金额：明细销售额之和减订单优惠，再加运费；
- 退款统计应明确是否只计算 `refunds.status = 'approved'`；
- 日期字段使用 `YYYY-MM-DD HH:MM:SS` 文本格式，可使用 SQLite 的 `strftime`。

## 开源说明

本项目基于 Apache-2.0 协议的 [MiniMind](https://github.com/jingyaogong/minimind) 进行扩展，保留原仓库许可证和归属说明。MiniSQL-RL 新增的电商数据层、SQL 沙箱、评测集及后续训练代码会在提交记录和文档中明确标注。
