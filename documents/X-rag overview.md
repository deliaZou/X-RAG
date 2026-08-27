## 总体设计

三层递进,每层输出作为下一层输入,信息逐层压缩:

```
原始指标(68维,上千时间点)
   ↓ Layer 1 检测
异常时刻与异常点
   ↓ Layer 2 归因
候选服务与指标证据(结构化)
   ↓ Layer 3 诊断
根因排序(service_metric列表)
```

---

## Layer 1:异常检测

**输入**:每案例的训练段(注入前 480 行,无标签)与测试段(720 行,含推导标签)。

**算法**:PCC(主成分重构),经八算法对比选定。判据从单一检测质量扩展为四维:检测质量、归因质量、阈值可迁移性、计算成本。核心发现——检测已饱和(多数算法 AUC-PR 逼近 0.99),归因质量取代检测质量成为决定性判据。

**阈值**:训练段分数的 q=0.999 分位数,五折折感知标定,一致选出。阈值只在训练数据上定,不碰测试数据。

**输出**:每个测试点的异常判定,标记异常段供 Layer 2 解释。

**已知局限**:高维系统(TT,353维)上阈值可迁移性显著下降(FPR 从 OB 的 0.16 升至 TT 的 0.76),主实验限定在 68 维的 OB。

---

## Layer 2:XAI 归因

**输入**:Layer 1 标记的异常点,原始 68 维指标。

**方法**:KernelSHAP 对 PCC 分数做归因,逐点计算后按特征聚合(绝对值均值),再按服务名汇总得到服务级排序。

**核心发现**:归因质量与 SHAP 精确性无关。TreeSHAP 精确但对应树模型归因命中率仅 0.04-0.06;KernelSHAP 近似但对应 PCC 等重构类方法命中率达 0.43。原因是 TreeSHAP 精确解释的隔离路径深度与根因无关且会饱和,KernelSHAP 近似解释的重构残差与根因结构对齐。

**输出契约**:每案例一份 JSON,含全部候选服务(按 SHAP 排序)及其全部指标(shap 值、方向),供下游按 Top-N 截断使用。评测元数据(case_id、ground_truth)单独存放,不进入下游输入。

**已知局限,已诊断**:

- socket 类系统性失效(AC@1=0)。fidelity 测试证明 SHAP 归因忠实反映 PCC 分数(fidelity 0.963,不低),问题在 PCC 分数本身未将 socket 指标识别为主因,而非归因方法失效。
- z-score(不依赖模型的幅值排序)在 socket 类完胜 PCC(AC@5: 1.0 对 0.4),两者候选取并集可将整体候选覆盖率从 0.878 提升至 0.978。此项作为后续消融,暂未并入主线。
- cpu、mem 类存在明显的"服务定位准、指标定位错"现象(cpu 粗粒度 0.8 对细粒度 0.133),即 deviation-cause misalignment:结果指标(latency/workload)的偏移幅度系统性大于根因指标(cpu/mem)。这是 Layer 3 的主要靶点。

---

## Layer 3:RAG-LLM 诊断(待实现)

**输入**:Layer 2 的候选证据(Top-N 服务及指标) + 检索到的领域知识。

**知识库**(静态,不含案例历史,故无需 fold 隔离):

- 指标语义卡(7类):cpu、mem、diskio、socket、latency、workload、error 各自的含义、故障表现、因果方向。
- 服务档案卡(11张):各服务职责与调用依赖关系。
- 类型级故障签名(6类):每种故障类型的候选粗/细粒度命中率、已知误判模式(如 cpu 类中 latency 常盖过 cpu)。

**任务**:在候选服务与指标内,依据因果知识(资源饱和是因,latency/workload/error 是果)重排序,输出补满 5 项的 service_metric 排序列表。

**评测**:细粒度 AC@k,对比 Layer 2 单层(总体 AC@1=0.433,分类型差异大)。目标是缩小 cpu、mem 类的粗细粒度缺口。

---

## 架构设计原则

1. **信息隔离**:LLM 与检索均不接触原始时序、标签、案例 ID、真实答案,仅接触 Layer 2 输出的结构化候选证据。
2. **知识库无泄漏**:全部为静态文档或类型级统计,不绑定具体服务身份,天然避免历史案例泄漏,无需交叉验证式隔离。
3. **逐层可独立评测**:检测轴(FPR)、归因轴(AC@k 对 nsigma)、诊断轴(AC@k 对 Layer2/基线)分别有对照,支持逐层消融。
4. **诚实披露已知局限**:socket 候选缺失、高维阈值失效均已定位机制并如实记录,不掩盖。


## 知识库文档，基于已确认信息整理

按之前定的方向：知识库不用原始 log，改用指标语义卡（静态）+ 服务档案卡（静态）+ 类型级故障签名（从 metrics 统计，不绑服务身份，不需要 fold 隔离）。

---

### 一、指标语义卡（7 张，静态，与案例数据无关）

**CPU**

- 含义：容器 CPU 使用率
- 故障时表现：饱和后处理时间上升，可能引发同服务 latency、workload 上升
- 因果方向：CPU 饱和是因，latency/workload 上升是果，反之不成立
- 传播模式：本服务内传播为 latency 上升；跨服务传播为下游调用方超时

**MEM**

- 含义：内存占用
- 故障时表现：持续爬升不回落（区别于 CPU 的瞬时波动），严重时触发 OOM
- 因果方向：内存泄漏是因，GC 频率上升、latency 抖动是果
- 传播模式：与 CPU 类似但爬升更平缓、更持续

**DISKIO**

- 含义：磁盘读写量
- 故障时表现：注入后指标幅度跃升数个量级（本数据集中最易识别，AC@1=1.0）
- 因果方向：磁盘 I/O 压力是因，落盘延迟、请求排队是果
- 已确认：本数据集中信号最强，无需额外知识辅助

**SOCKET**

- 含义：网络连接数/套接字压力
- 故障时表现：原始偏移极大（zscore 排名第一），但 PCC 重构残差不敏感
- **已确认局限**：PCC 归因系统性失效（AC@1=0），fidelity 测试证明非 SHAP 不忠实，而是 PCC 分数本身未将其识别为主因；需 zscore 作为候选补充
- 因果方向：socket 压力是因，连接超时、error 计数是果

**LATENCY**

- 含义：请求响应延迟（P90）
- 故障时表现：几乎所有故障类型下都会跟随上升，是最常见的"结果指标"
- **已确认关键点**：latency 的偏移幅度常年大于真正根因指标（cpu、mem），是 deviation-cause misalignment 的主要载体
- 因果方向：几乎不作为原因，几乎总是结果

**WORKLOAD**

- 含义：请求吞吐量/负载
- 故障时表现：随根因指标同步变化，常见于 delay/loss 类
- 因果方向：多为结果指标，随上游资源瓶颈被动变化

**ERROR**

- 含义：错误请求计数
- 故障时表现：稀疏指标，仅部分服务持续产生（如 frontend-external 83/90 案例），多数服务无 error 记录
- 因果方向：结果指标，反映故障已传播至请求失败层面

---

### 二、服务档案卡（11 张，静态，来自系统架构）

|服务|职责|上下游关系|
|---|---|---|
|frontend|用户请求入口|调用几乎所有其他服务；下游任何服务故障都可能反映在其 latency 上|
|checkoutservice|结算流程编排|调用 payment、currency、cart、shipping；本身故障会导致这些服务的调用方超时|
|cartservice|购物车管理|依赖 redis|
|currencyservice|货币转换|被 checkoutservice、frontend 调用|
|paymentservice|支付处理|被 checkoutservice 调用|
|shippingservice|配送报价|被 checkoutservice 调用|
|productcatalogservice|商品目录|被 frontend、recommendationservice、checkoutservice 调用|
|recommendationservice|推荐引擎|调用 productcatalogservice|
|adservice|广告投放|相对独立，下游依赖少|
|emailservice|订单确认邮件|被 checkoutservice 调用，异步|
|redis|购物车缓存|被 cartservice 依赖|

**已确认**：这 11 个服务是列空间的白名单来源，排除的是 istio-init、PassthroughCluster、frontend-check、frontend-external 等服务网格组件（非应用服务，不可能是根因）。实际会被注入故障的只有 5 个：checkoutservice、currencyservice、emailservice、productcatalogservice、recommendationservice。

---

### 三、类型级故障签名（6 张，从训练数据统计，类型级不绑服务身份，故不需 fold 隔离）

**CPU 故障签名**（15 案例统计）

- 根因服务粗粒度命中率 0.800，细粒度仅 0.133——**几乎所有 cpu 案例中，latency 或 mem 的偏移超过 cpu 本身**
- 判别规则：若候选服务内 cpu 与 latency 同时显著异常，且无其他解释，优先判 cpu 为因

**MEM 故障签名**

- 粗粒度 0.933，细粒度 0.467，同 cpu 类模式，缺口略小
- 表现：爬升曲线较 cpu 更平缓持续

**DISK 故障签名**

- 粗细粒度均 1.000（或接近），无需特殊规则，直接采纳幅值最大的指标

**SOCKET 故障签名**

- **已确认局限**：PCC 候选集漏检率高（60% 案例根因掉出 Top5），需并入 zscore 候选补充，Layer 3 重排序无法弥补候选缺失
- 若根因指标已在候选内（40% 案例排位 4-5），可用判别规则提升

**DELAY / LOSS 故障签名**（latency 族故障）

- 粗粒度 0.6-0.667，细粒度 0.467-0.533，缺口中等
- 表现：workload 与 latency 常同步剧烈波动，需要拓扑知识（谁调用谁）辅助判断源头服务

---

### 四、核心因果规则（供 Layer 3 提示模板复用）

1. 资源饱和（cpu/mem/diskio/socket）是因，latency/workload/error 几乎总是果。
2. latency 偏移幅度系统性大于其根因指标，不能按幅值直接判定。
3. 若候选服务的资源类指标（cpu/mem/diskio/socket）与结果类指标（latency/workload/error）同时异常，优先判资源类指标为根因。
4. socket 需并入 zscore 候选，PCC 候选不完整。
5. disk 类无需额外规则。

--- 

### 五、待后续补充（非静态，等结果确认后再定）

- zscore 与 SHAP 融合权重（消融后确定）
- LLM 输出格式模板（Layer 3 跑通后固化）
- fidelity 是否作为案例级提示（当前结论：不加）