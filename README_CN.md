# OpenLLM

[中文说明](#中文说明) | [English Description](./README.md)

---

## 中文说明

OpenLLM 是一个用于 Go 语言的轻量级、统一的大语言模型（LLM）集成包。它旨在通过统一的接口封装不同 LLM 提供商（如 OpenAI 和 Anthropic Claude）的差异，为开发者提供一致的使用体验。

### 核心特性

- **统一接口**：一个 `Model` 接口涵盖所有支持的 LLM，切换模型只需更改初始化逻辑。
- **完整支持**：支持阻塞式对话（Chat Completion）和流式输出（Streaming）。
- **智能工具调用**：
  - 支持函数调用（Function Calling）。
  - **自动解析**：通过 Go 反射机制，自动从处理函数的参数结构体中解析 JSON Schema（支持自定义标签 `openllm`）。
- **消息序列化**：提供跨提供商的消息编解码工具，方便对话历史的持久化与恢复。
- **详尽统计**：在响应中包含 Token 消耗（输入/输出/缓存命中）、请求耗时及元信息（RequestID, Fingerprint 等）。
- **多模态支持**：支持图片输入（OpenAI 路径）。

### 安装

```bash
go get github.com/thecxx/openllm
```

### 快速开始

#### 1. 初始化模型

**OpenAI**
```go
import (
    "github.com/thecxx/openllm"
    "github.com/sashabaranov/go-openai"
)

client := openai.NewClient("your-api-key")
model := openllm.NewLLM("gpt-4o", "OpenAI GPT-4o", client)
```

**Anthropic Claude**
```go
import "github.com/thecxx/openllm"

// 使用 API Key 直接初始化
model := openllm.NewAnthropicLLMWithAPIKey("claude-3-5-sonnet-20240620", "Claude 3.5 Sonnet", "your-api-key")
```

#### 2. 对话调用

```go
ctx := context.Background()
messages := []openllm.Message{
    model.NewUserMessage("你好，请介绍一下你自己。"),
}

resp, err := model.ChatCompletion(ctx, messages)
if err != nil {
    log.Fatal(err)
}

fmt.Println("回答:", resp.Answer().Content())
fmt.Printf("消耗: %+v\n", resp.Usage())
```

#### 3. 流式对话

```go
// 实现 StreamWatcher 接口
watcher := &MyStreamWatcher{} 

resp, err := model.ChatCompletionStream(ctx, messages, openllm.WithStreamWatcher(watcher))
```

#### 4. 自动解析函数工具 (Tool Calling)

你可以定义一个普通的 Go 函数，并通过反射自动生成工具定义：

```go
type SearchParams struct {
    Query string `openllm:"query,required,desc=搜索关键词"`
    Limit int    `openllm:"limit,desc=返回结果数量"`
}

func Search(ctx context.Context, params *SearchParams) (string, error) {
    return fmt.Sprintf("搜索结果: %s", params.Query), nil
}

// 自动解析参数并定义工具
tool := openllm.DefineFunction(
    "search_engine", 
    "在互联网上搜索信息",
    openllm.WithInvokeFunc(Search),
)

resp, err := model.ChatCompletion(ctx, messages, openllm.WithTools([]openllm.Tool{tool}))
```

#### 5. 消息持久化 (序列化)

由于不同模型的内部消息结构不同，OpenLLM 提供了统一的序列化方案：

```go
// 序列化为 JSON
data, err := openllm.EncodeMessage(resp.Answer())

// 从 JSON 反序列化并恢复给特定模型使用
// 即使 model 切换了（如从 OpenAI 换到 Claude），反序列化也会自动适配
restoredMsg, err := openllm.DecodeMessage(model, data)
```

### 项目结构

- `model.go`: 定义核心 `Model` 接口。
- `openai.go` / `anthropic.go`: 各提供商的具体实现。
- `define.go`: 工具与函数定义的通用逻辑。
- `template.go`: 基于反射的参数解析模版。
- `message.go`: 消息接口与序列化工具。
- `response.go`: 响应接口与统计结构。

### 开源协议

Apache License 2.0
