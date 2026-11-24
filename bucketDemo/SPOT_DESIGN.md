# bucket2 spot instance 中断 / 恢复 — 设计 (branch `spot/bucket2`)

用户决策 (2026-11-20 / 21)，所有实现必须遵守:

| 项 | 决定 |
|---|---|
| 目标规模 | 1B；当前测试规模 SIFT10M (正确性)、SIFT100M (开销)；1B 只做设计保证 (int64 游标、单元按全局 id) |
| 哪些阶段用 spot | **Step 4、Step 6、prune**。optimize: 不用 spot、不用多卡 (spot 模式下 `--num-gpus 1`，跑在 on-demand 卡上) |
| 部署形态 | **本机模拟**: 本机每张 GPU = 一个逻辑 spot 实例；外部 `spot_sim.py` 回放 trace，发中断信号，GPU worker 线程收到后停止 / 恢复 |
| 串行步骤 | Step 1/2/3/4.5/5、merge、4d 在 on-demand 卡上 (随便一张空闲卡，标为不会被抢占 = CUDA 可见设备 0) |
| on-demand 卡在并行阶段 | **一直参与，但最后才领、只领小段** (吞吐最高)；spot worker 优先 |
| 换机型 / 换 GPU | 允许；单元按全局编号定义，每个 worker 按自己的显存算批大小 / 缓存 |
| 结果 | 不要求跟不中断时逐字节相同，但**算法正确性不能受影响**。确定性模式下 (BUCKET2_ASSIGN_DETERMINISTIC=1) 有抢占的结果必须跟没抢占的逐字节相同 (这三个阶段的每个单元都是确定的，所以做得到) |
| 恢复语义 | 记录每个 worker 执行到哪里，恢复后从中断位置继续。bucket 按 bucket order 顺序执行 → 一个游标 = "这之前全做完了"；多 GPU → 多个游标 |
| trace | 用户的真实 trace 是 **Bamboo**；目前用合成信号 (`spot_sim.py gen`)。时间按比例压缩 (`--scale`，默认 60: 1 real s = 1 trace min)；以后在 1B 上用原始间隔 |

## 1. 架构

```
           host (常驻, 不会被抢占)                                   spot_sim.py (外部进程)
  ┌───────────────────────────────────────────────┐                 trace CSV → 压缩时钟
  │ SegmentScheduler (每个 stage 一个)              │   mmap 状态文件   │
  │   Segment{begin, cursor, end, owner, issued}   │◄──── state.bin ──┤ notice / preempt / join
  │   commit: 验证 owner + 未过抢占时刻 → 写结果 →   │   SIGUSR1        │
  │           推进 cursor                            │◄─────────────────┘
  │ host 端结果: h_top2_all / arrival 累加器 / 输出文件│
  └───────▲──────────────▲──────────────▲──────────┘
          │ commit       │              │
     worker 0 (dev 0)  worker 1 (dev 1)  worker 2 (dev 2) ...   (每张卡一个线程; dev 0 = on-demand)
```

- host 不会挂 → **worker 被抢占时不需要把状态写盘**。结果交回 host 就算提交；丢的只是在途单元和那个实例的 GPU 缓存。
- 模拟上，"实例没了" = worker 线程在下一个检查点停下、丢弃没提交的在途结果、释放显存 (teardown)，然后等 trace 里的 `join` (新实例，epoch+1，冷启动 setup)。
- **抢占时刻之后才到的提交一律拒收** (`commit` 检查 `now < deadline`)。所以通知只是优化 (少丢在途工作)，不是正确性前提；没有通知的硬抢占同样正确。

## 2. 实例 ↔ 设备 ↔ worker

`spot::make_workers(G)` (src/spot_sched.hpp):

| 模式 (`BUCKET2_SPOT_MODE`) | 默认用于 | 设备 0 | 设备 1..G-1 | trace 里的实例 |
|---|---|---|---|---|
| `dedicated` | G ≥ 2 | on-demand worker (参与，最后领、领小段；不会被抢占) | spot 实例 1..G-1 | 1..G-1 |
| `shared` | G = 1 | spot 实例 0 **+** 同卡的 on-demand 救场 worker (所有 spot worker 都不可用超过 `BUCKET2_SPOT_RESCUE_S` 才接活；懒 setup，不干活不占显存) | spot 实例 1..G-1 | 0..G-1 |

串行步骤永远在设备 0 (on-demand 角色)。没有 `BUCKET2_SPOT_DIR` → spot 关闭: 每个设备一个普通 worker、静态切分、不偷 = 原来的行为。

## 3. 段调度器 (src/spot_sched.hpp，已实现 + 单元测试)

- `SegmentScheduler(stage, n_units, workers, init_cuts, init_owner, opt, unit_cost)`。初始切分 = 原来的静态切分 (Step 6 的代价切分、Step 4 的行切分、prune 的点数切分)，每段**预留**给对应 worker (它第一次 acquire 时领走；它的实例一开始就不可用 → 别人可以拿)。
- `run_worker(w, setup, proc, teardown)`: 等实例 Up → 领段 → (第一次领到时) setup → `proc(L)` → release → 被抢占就 teardown 并等新实例 (epoch 变大)。
- `proc(L)` 里的协议:
  ```cpp
  for (u = L.begin; ; u = u1) {
      u1 = 下一个单元的末尾 (≤ S.end_of(L))
      if (u >= S.end_of(L) || !S.may_start(L, u, u1)) break;   // 被偷到这里 / 被抢占 / 通知期来不及
      ...发出 [u, u1) 的计算 (可以流水线: 多个单元在途)...
      // 计算完成 (按顺序) 时:
      S.commit(L, u1, [&]{ 把 [u, u1) 的结果写进 host 结构 });       // false = 拒收，丢弃结果
  }
  // 把已发出的在途单元做完 / 提交 (或被拒收) 后返回
  ```
- `may_start` 记录 `issued` (已发出的最远位置)；偷的切点 ≥ issued，所以偷不会浪费原 owner 的在途工作。通知期内: 预计完成时刻 (在途单元数 × 每单元 EWMA) + `drain_margin_s` 超过 deadline 就停 (drain)。
- `commit(L, new_cursor, fn)`: 段的提交锁内校验 (还是 owner、token 对、实例没过抢占时刻、单调)，然后执行 `fn` 并推进 cursor。不同段的提交并行 (每段自己的锁) → Step 6 的每卡 lane 仍并行。
- 回收 (reap): acquire 时发现某段 owner 的实例已 Down → 立刻把 `[cursor, end)` 放回空闲 (模拟 host 发现实例没了)。
- 空闲 worker: 预留段 > 空闲段 (起点最小，保持 bucket order 局部性) > 偷 (剩余代价最大的段，从 issued 之后按代价对半切)。on-demand: spot 有人在等就让；每次最多拿 `BUCKET2_SPOT_OD_SHARE` × 平均份额。
- 统计: `print_stats` 打 `[spot] stage=... units= segments= steals= drains= reaped= preempts= rejected_units= setups= wall=` + 每个 worker 一行；run_pipeline 的 summarize.py 收进 metrics.json `spot_reps`。
- 测试: `src/tests/test_spot_sched.cpp` (7 个用例: 关闭 / dedicated / 流水线 / G=2 / shared G=1 救场 / 硬抢占 / 空; 每单元恰好提交一次 + 覆盖完整; TSan 干净)，`src/tests/test_spot_e2e.cpp` (真实 spot_sim.py + mmap + SIGUSR1，跨进程)。

## 4. 各阶段接入 (loop 要做的事)

所有接入都是: spot 开启时 (`spot::Client::get().enabled()`) 走调度器；关闭时代码路径/行为不变 (或者调度器在关闭模式下与原静态切分逐字节等价)。

### 4.1 Step 4 (`batch_assign_with_cagra_anns`, bucket.cu ~L1700–2520)
- 单元 = non-centroid 行 `[0, N_nc)`；初始切分 = 现在的 `N_nc*g/G4`。
- `top2_worker` 拆成 setup (CAGRA 索引 / uint8 副本 / buffer 分配; 需要的 `d_centroids` / `d_graph` 在 setup 里 peer-copy 到本卡) / proc (批循环从 `L.begin` 开始，`batch_end_of` 不能越过 `S.end_of(L)`，每批前 `may_start(L, batch_start, batch_end)`，批完成后 `commit(L, batch_end, memcpy 进 h_top2_all)`) / teardown (释放)。prefetch_io 的预读要能在 may_start=false 时取消/等待干净。
- 正确性: 每行 top-2 跟在哪张卡、批里第几个无关 (现有注释已说明; 确定性模式逐字节相同)。4d 在 host 上、所有单元完成之后，按 `Nv_assign` 批次，不受影响。
- 抢占代价 = 在途那一批 (SIFT100M 每批 ~8 s / GPU) → 优化方向: spot 模式下更小的提交粒度。

### 4.2 Step 6 (`run_step6` lambda bucket.cu ~L10140 → `build_vector_knn_with_tensorcore(..., pos_begin, pos_end, acc_lane)`)
- 单元 = bucket order 位置 `[0, npos)`；初始切分 = 现在的 S3 代价切分 (cut)，`unit_cost` = wcost (偷时按代价对半)。
- 每个 ScatterJob 是一个 bucket；需要带上它的 pos。提交点 = scatter 线程按 seq 顺序把这个 bucket 的记录 append 进 arrival lane 的时候: `commit(L, pos+1, append_encoded 循环)`。被拒收的 job 直接丢掉 (不 append)。
- 主循环 (L8206 `for pos`) 每个 bucket 前 `may_start(L, pos, pos+1)`；false → 停止发新 bucket，等 slot / encoder / scatter 排空后返回。`future_access` / Belady 缓存按租约的 `[L.begin, end)` 建 (偷会让 end 变小: 多出来的 future 信息只是变旧，不影响正确性；越界检查见 scale 不变式 4)。
- 每张卡一条 lane (S2): lane 号 = worker 下标 (`acc.set_lanes(workers.size())`)。被抢占的 worker 的 lane 里已经 append 的都是已提交的完整 bucket。
- 正确性: bucket 互不相交，每轮每点一条记录；merge 结果跟记录到达顺序无关；kth seed 是上界 (哪张卡用都对)。

### 4.3 prune (`prune_windowed.cpp` L573 多 GPU 分支 → `prune_windowed_gpu::run_chunked(..., pos_lo, pos_hi)`)
- 单元 = bucket 位置 `[0, n_buckets)`；初始切分 = 现在的按点数切分。
- `run_chunked` 要支持: 动态 end (每块前 `may_start(L, pl.p0, pl.p1)`；块规划 `plans` 可以在 end 变小时截断)，以及块 c 的结果 `finish(c)` 写出后 `commit(L, pl.p1, ...)`。写出是按偏移 pwrite (`write_rows_at`)，被拒收的块不写 (或写了也无妨: 重做的块写同样的字节——确定性)。
- 正确性: 每个点只看自己的窗口，结果确定。

### 4.4 optimize_chunked
不接入。spot 模式下 run_pipeline 给它 `--num-gpus 1` (on-demand 卡)。

## 5. 模拟器与测试工具

- `.claude/skills/optimize-bucket2-spot/scripts/spot_sim.py`:
  - `gen --instances 1,2,3 --mtti-s --down-s --notice-s --hard-frac --correlated --seed -o trace.csv` (突发模型: 全局泊松抢占事件，30% 一次回收多台; 20% 无通知)
  - `run --trace --dir --n-inst G --scale 60 [--notice-real-s X] --log`: 写 `<dir>/state.bin`，每个事件给 `<dir>/pids/` 里登记的进程发 SIGUSR1
  - `show --dir`
- 参考 trace (`traces/`): `t1_g{1,2,4}.csv` (mtti 1 h、替补 ~10 min、通知 120 s、20% 硬抢占、30% 关联; scale 60 下 SIFT100M 一次 build 约 4 trace 小时 → G=4 约 20 次抢占)，`d1_g{1,2,4}.csv` (高密度，正确性用)。
- run_pipeline.sh (本分支): `SPOT=1 SPOT_TRACE=... SPOT_TAG=t1 SPOT_SCALE=60 MGPU=G` → 每个 rep 在 bucket2 前启动 spot_sim，prune 后停止；`spot_events.log` 进 rep 目录；judge key `<ds>_<mode>_g<G>_spot<TAG>`。

## 6. 指标

- **正确性 (硬门槛)**: 确定性模式下 spot + trace 的 `vector_knn.bin` / `forward_graph_reordered.bin` / `final_graph.bin` 跟同 G 无 spot 的确定性运行逐字节相同 (`spot_det.sh`)；正常模式 recall/QPS 过 judge 质量门；0 图缺陷。
- **无事件开销**: `SPOT=1` 无 trace 的总时间 vs 同期非 spot HEAD ≤ +3%。
- **抢占下的效率**: 同一 trace 下的总时间 (judge key `_spot<TAG>`)，以及 lost work = rejected_units 折算秒、冷启动 setup 秒、drain 次数、偷的次数。优化目标: 减少 "trace 下总时间 − 无事件总时间"。
