# 如何运行测试脚本

## 快速开始（推荐）

```bash
cd /home/lanlu/microBenchmark/bucketDemo/buildBucket
bash quick_test.sh
```

这将：
1. 生成100K向量的测试数据
2. 编译项目
3. 运行一个简单的测试

## 详细测试（包含多个场景）

```bash
cd /home/lanlu/microBenchmark/bucketDemo/buildBucket
bash run_test.sh
```

这将运行3个测试：
- **Test 1**: 小数据集(10K向量) - 直接在GPU中处理，无需采样
- **Test 2**: 中型数据集(100K向量) - 使用采样
- **Test 3**: 大数据集(1M向量) - 使用PQ压缩

## 手动编译

```bash
cd /home/lanlu/microBenchmark/bucketDemo/buildBucket/build
cmake ..
make -j$(nproc)
```

## 手动运行（使用现有数据）

```bash
cd /home/lanlu/microBenchmark/bucketDemo/buildBucket

# 第一次需要生成测试数据
python3 generate_test_data.py

# 然后运行bucket可执行文件
./build/bucket \
    --data test_data/vectors_100k_128d.fbin \
    --k 128 \
    --m 32 \
    --out_dir ./output/my_test \
    --seed 42 \
    --gpu-limit $((2 * 1024 * 1024 * 1024)) \
    --sample-rate 0.1 \
    --centroid-ratio 0.01 \
    --pq-bits-start 8
```

## 命令行参数说明

### 必需参数
- `--data`: 输入数据文件路径 (.fbin/.ibin/.u8bin/.i8bin格式)
- `--k`: 桶数
- `--m`: 每个点的邻近点数
- `--out_dir`: 输出目录

### GPU内存优化参数（我们修改的部分）
- `--gpu-limit`: GPU内存限制(字节)，默认4GB
- `--sample-rate`: 采样比例 [0, 1]，默认0.1 (10%)
- `--centroid-ratio`: 聚心点比例，默认0.01 (1%)
- `--pq-bits-start`: PQ起始比特数，默认8 **[不再限制在4-8]**
- `--pq-bits-min`: PQ最小比特数，默认4 **[不再限制在4-8]**
- `--use-pq`: 强制使用PQ量化

### 其他参数
- `--seed`: 随机种子，默认0
- `--kmeans-iters`: KMeans迭代次数，默认5
- `--init-method`: 初始化方法 (kmeans/kmeans-fast/random)
- `--no-balance`: 禁用桶均衡
- `--balance-slack`: 均衡松弛度，默认4
- `--t`: 轮次数，默认1

## 关键改进说明

### 1. PQ比特数无限制
原先：`pq_bits` 被限制在 [4, 8]
现在：可以设置任意值 >= 4，甚至 1 比特

**使用建议**：
```bash
# 激进压缩
--pq-bits-start 8 --pq-bits-min 1
```

### 2. 快速采样策略
原先：Fisher-Yates shuffle (O(N) 初始化 + O(n_samples) 交换)
现在：直接随机生成 (O(n_samples) 快速随机数)

**好处**：采样速度提高，尤其在 `sample_rate` 较小时

### 3. 自适应PQ维度
当 `pq_bits` 降到最小值仍无法fit GPU时：
- 增加 `pq_dim` (减少子空间数)
- 进一步压缩编码数据
- 自动扩展搜索范围直到找到可行配置

**流程**：
```
pq_bits: 8 → 7 → 6 → ... → 1
         ↓ (如果仍不fit)
pq_dim:  初始 → 初始*2 → 初始*4 → ...
         (每个pq_dim，重新尝试所有pq_bits)
```

## 测试数据说明

脚本会生成3个测试数据集：

| 文件 | 向量数 | 维度 | 文件大小 |
|------|--------|------|----------|
| vectors_10k_128d.fbin | 10,000 | 128 | ~5 MB |
| vectors_100k_128d.fbin | 100,000 | 128 | ~50 MB |
| vectors_1m_128d.fbin | 1,000,000 | 128 | ~500 MB |

## 常见场景

### 场景1: 小数据集，无需优化
```bash
./build/bucket \
    --data test_data/vectors_10k_128d.fbin \
    --k 32 --m 32 \
    --out_dir output/small \
    --sample-rate 1.0
```

### 场景2: 中等数据集，采样处理
```bash
./build/bucket \
    --data test_data/vectors_100k_128d.fbin \
    --k 128 --m 32 \
    --out_dir output/medium \
    --sample-rate 0.2 \
    --centroid-ratio 0.01
```

### 场景3: 大数据集，激进PQ压缩
```bash
./build/bucket \
    --data test_data/vectors_1m_128d.fbin \
    --k 256 --m 32 \
    --out_dir output/large \
    --gpu-limit $((2 * 1024 * 1024 * 1024)) \
    --sample-rate 0.05 \
    --pq-bits-start 8 \
    --pq-bits-min 1 \
    --use-pq
```

## 输出解释

运行完成后会生成：
```
output/
├── vectors_norm.bin          # 归一化向量
├── bucket_assignment.bin     # 桶分配信息
├── bucket_centroids.bin      # 桶心信息
└── graph.bin                 # 构建的图结构
```

## 故障排除

### 1. 编译失败
检查依赖：
```bash
# conda环境激活
conda activate rapids_raft

# 检查CUDA、RAFT等
which nvcc
echo $CONDA_PREFIX
```

### 2. GPU内存不足
减小 `--gpu-limit` 或 `--sample-rate`：
```bash
--gpu-limit $((1 * 1024 * 1024 * 1024))  # 1GB
--sample-rate 0.05                        # 5%
```

### 3. 运行时报错
查看详细日志，通常会输出内存估算信息帮助诊断。

## 性能测试建议

使用 `time` 或 `/usr/bin/time -v` 进行性能测试：

```bash
/usr/bin/time -v ./build/bucket \
    --data test_data/vectors_100k_128d.fbin \
    --k 128 --m 32 \
    --out_dir output/perf_test \
    --sample-rate 0.1
```

这会显示：
- 执行时间
- 内存使用峰值
- CPU使用率
- I/O统计

