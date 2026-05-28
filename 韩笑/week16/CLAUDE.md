# 多智能体狼人杀 Team 系统 - CLAUDE.md

## 项目概述

基于大语言模型（LLM）的多智能体狼人杀 Team 系统。每个 Agent 代表一个独立玩家，具备高度自主决策与博弈对抗能力。系统采用 **Server-Authoritative（服务端权威）** 架构，所有 Agent 只与 FastAPI 后端通信，由后端 Game Master 负责信息过滤、规则校验和消息分发。

## 技术栈

| 组件 | 技术选型 | 用途 |
|------|----------|------|
| Web 框架 | FastAPI | 原生 async/await，Pydantic V2 校验 |
| Agent 编排 | LangGraph | 内置状态机和消息路由 |
| 实时通信 | WebSocket | 强实时交互，低延迟 |
| 数据存储 | SQLite | 单文件存储对局记录和 Agent 记忆 |
| LLM 调用 | DeepSeek API (OpenAI SDK) + LiteLLM | 在线推理 deepseek-v4-pro + 本地模型兼容 |
| 任务队列 | asyncio.Queue | 本地单机任务调度 |

## 核心设计原则

### 1. Server-Authoritative（服务端权威）

AI Agent **绝不直接互相调用接口**。所有 Agent 只与 FastAPI 后端通信，由后端充当"裁判（Game Master）"负责信息过滤、规则校验和消息分发。从根本上杜绝信息泄露和规则作弊。

### 2. 信息隔离架构

采用 **"中央裁判 + 私有信道"** 架构：

```
Game Master / 裁判系统
├── 公开广播 → 公共频道（发言/投票/死讯）→ 所有 Agent
├── 私有消息 → 狼人私有频道（队友/击杀）→ 仅狼人 Agent
├── 私有消息 → 预言家私有频道（验人结果）→ 仅预言家 Agent
└── 私有消息 → 女巫私有频道（被刀者/药效）→ 仅女巫 Agent
```

### 3. 全程可观测性

系统实时输出结构化日志，记录 Agent 的：
- 内心独白（Thought/Chain-of-Thought）
- 发言草稿
- 策略修正过程
- 投票决策依据

## 游戏规则

### 阵营与胜利条件（屠边规则）

| 阵营 | 身份 | 胜利条件 |
|------|------|----------|
| 好人阵营 | 预言家、女巫、猎人、村民 | 消灭所有狼人 |
| 狼人阵营 | 基础狼人 | 所有神职全灭 **或** 所有平民全灭 |

### 游戏阶段流转

```
天黑闭眼 → 狼人密聊刀人 → 预言家验人 → 女巫用药判定 → 天亮了
                                                              ↓
公投放逐 ← 自由发言辩论 ← 竞选警长(上警) ← 法官宣布死讯 ← 结算夜间伤亡
```

### step 阶段索引

| 索引 | 阶段标识 | 说明 |
|------|----------|------|
| 0 | `night_wolf` | 狼人投票选择击杀目标 |
| 1 | `night_seer` | 预言家查验一名玩家 |
| 2 | `night_witch` | 女巫决定是否用药 |
| 3 | `night_result` | 宣布夜晚死亡结果 + 猎人开枪结算 |
| 4 | `day_start` | 白天开始，公布昨晚信息 |
| 5 | `speech` | 所有存活玩家依次发言 |
| 6 | `vote` | 全体投票放逐 |
| 7 | `day_end` | 检查胜负 → 天数+1 或游戏结束 |

## 角色设计

### 狼人阵营（2人）

| 角色 | 已知信息 | 未知信息 | 开发关键点 |
|------|----------|----------|------------|
| 普通狼人 | 队友编号；每晚击杀目标；战术沟通 | 所有神职/平民身份；预言家验人；女巫用药 | 团队共识机制；内部私信；意见不合时兜底策略（空刀/随机）；Prompt 含欺骗策略和角色扮演指令 |

### 神职阵营

| 角色 | 已知信息 | 未知信息 | 开发关键点 |
|------|----------|----------|------------|
| 预言家 | 查验过的玩家身份（累积记忆） | 未查验玩家身份；其他神职；狼人具体身份 | 核心是记忆模块：维护 VerifiedList，防止遗忘或篡改历史验人结果 |
| 女巫 | 每晚被刀玩家；双药使用状态 | 被刀者真实身份；狼人身份；验人结果 | 资源状态锁（hasAntidote, hasPoison）；首夜自救规则可配置；6人局建议仅保留解药 |

### 平民阵营（2人）

| 角色 | 已知信息 | 未知信息 | 开发关键点 |
|------|----------|----------|------------|
| 平民 | 仅知道自己身份 | 一切：狼人、神职、夜晚信息 | Prompt 强调"逻辑分析"而非"信息陈述"；注入表水和站边逻辑模板；禁止编造信息 |

## 目录结构

```
werewolf-api/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── ws/                  # WebSocket 管理器
│   │   └── connection.py
│   ├── game/                # 纯逻辑层（不依赖 FastAPI）
│   │   ├── game_master.py   # 裁判核心
│   │   ├── state_machine.py # 阶段流转
│   │   └── boards/          # 板子配置 JSON
│   ├── agents/              # Agent 适配层
│   │   ├── base_agent.py
│   │   └── llm_adapter.py   # LiteLLM 封装
│   └── schemas/             # Pydantic 消息模型
├── prompts/                 # 各角色 Prompt 模板（YAML/Jinja2）
├── tests/                   # 单元测试（重点测信息隔离）
└── docs/
    └── architecture.md      # 系统架构说明
```

## API 设计

### RESTful API（游戏生命周期管理，不参与回合内实时交互）

- `POST /api/v1/games` — 创建对局
- `GET /api/v1/games/{game_id}/state` — 获取游戏状态（仅返回请求者可⻅的公开信息）
- `GET /api/v1/games/{game_id}/replay` — 获取对局回放/日志
- `POST /api/v1/games/{game_id}/step` — 推进游戏阶段
- `POST /api/v1/config/reload` — 热重载 Prompt 配置

### WebSocket 接口（核心实时交互）

连接端点：`ws://localhost:8000/ws/{game_id}/{player_id}?token=xxx`

#### 服务端 → 客户端（下行消息）

| type | 触发时机 | 信息隔离 |
|------|----------|----------|
| `game_start` | 游戏开始 | 仅本人可⻅ role，狼人额外收到 teammates |
| `phase_change` | 阶段切换 | 所有人收到 |
| `private_info` | 私密信息推送 | 仅特定角色收到 |
| `public_broadcast` | 公开事件 | 所有人收到 |
| `action_request` | 要求 Agent 决策 | 仅当前行动角色收到 |
| `game_over` | 游戏结束 | 所有人收到 |

#### 客户端 → 服务端（上行消息）

| type | 触发时机 | 服务端校验 |
|------|----------|------------|
| `ready` | 收到 game_start 后 | 全员 ready 后才进入第一阶段 |
| `action` | 响应 action_request | 校验存活状态、目标合法性、阶段匹配 |
| `speak` | 白天发言阶段 | 校验发言顺序、内容长度 |
| `self_destruct` | 狼人自爆 | 校验狼人身份、白天发言阶段 |

## AI Agent 特殊优化

1. **超时兜底**：action_request 必须带 deadline，超时后 GM 自动执行默认动作（狼人空刀、神职跳过、平民弃票），绝不阻塞整局游戏
2. **结构化输出强制**：System Prompt 要求 Agent 以 JSON 回复，服务端二次解析；解析失败给一次重试，仍失败走兜底
3. **Token 限流**：GM 层加 `asyncio.Semaphore(2)` 限制同时推理的 Agent 数量，避免 OOM
4. **热重载**：Prompt 模板外置为 YAML/Jinja2，通过 config/reload 接口热更新

## 开发规范

- 所有消息体使用 Pydantic V2 模型严格校验
- WebSocket 消息采用 Discriminated Union 根据 type 字段路由到不同处理器
- 游戏逻辑层（game/）不依赖 FastAPI，保持纯 Python 可测试
- 信息隔离是测试重点：验证狼人收不到预言家验人消息、平民看不到夜晚信息等
- Agent Prompt 一律外置到 prompts/ 目录，不在代码中硬编码
- 阶段流转使用 LangGraph 状态机，避免手写 if-else 分支
