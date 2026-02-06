# OpenLLM

[English Description](#english-description) | [中文说明](./README_CN.md)

---

## English Description

OpenLLM is a lightweight, unified Large Language Model (LLM) integration package for Go. It simplifies the integration of different LLM providers (like OpenAI and Anthropic Claude) by providing a consistent interface.

### Core Features

- **Unified Interface**: Use a single `Model` interface for all supported LLMs. Switching providers only requires changing the initialization.
- **Full Support**: Supports both blocking Chat Completion and Streaming output.
- **Smart Tool Calling**:
  - Supports Function Calling.
  - **Auto Parsing**: Automatically generates JSON Schema from Go function parameter structs using reflection (supports custom `openllm` tags).
- **Message Serialization**: Provides cross-provider message encoding/decoding for easy conversation history persistence and restoration.
- **Detailed Statistics**: Includes token usage (input/output/cache hits), request duration, and metadata (RequestID, Fingerprint, etc.) in responses.
- **Multimodal Support**: Supports image inputs (OpenAI path).

### Installation

```bash
go get github.com/thecxx/openllm
```

### Quick Start

#### 1. Initialize Model

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

// Initialize with API Key directly
model := openllm.NewAnthropicLLMWithAPIKey("claude-3-5-sonnet-20240620", "Claude 3.5 Sonnet", "your-api-key")
```

#### 2. Chat Completion

```go
ctx := context.Background()
messages := []openllm.Message{
    model.NewUserMessage("Hello, please introduce yourself."),
}

resp, err := model.ChatCompletion(ctx, messages)
if err != nil {
    log.Fatal(err)
}

fmt.Println("Answer:", resp.Answer().Content())
fmt.Printf("Stats: %+v\n", resp.Stats().Usage)
```

#### 3. Streaming

```go
// Implement StreamWatcher interface
watcher := &MyStreamWatcher{} 

resp, err := model.ChatCompletionStream(ctx, messages, openllm.WithStreamWatcher(watcher))
```

#### 4. Auto Tool Parsing (Tool Calling)

Define a Go function and automatically generate the tool definition:

```go
type SearchParams struct {
    Query string `openllm:"query,required,desc=Search query keywords"`
    Limit int    `openllm:"limit,desc=Number of results to return"`
}

func Search(ctx context.Context, params *SearchParams) (string, error) {
    return fmt.Sprintf("Search results for: %s", params.Query), nil
}

// Auto-parse parameters and define tool
tool := openllm.DefineFunction(
    "search_engine", 
    "Search information on the internet",
    openllm.WithInvokeFunc(Search),
)

resp, err := model.ChatCompletion(ctx, messages, openllm.WithTools([]openllm.Tool{tool}))
```

#### 5. Message Persistence (Serialization)

```go
// Serialize to JSON
data, err := openllm.EncodeMessage(resp.Answer())

// Restore for a specific model (auto-adapts even if providers switch)
restoredMsg, err := openllm.DecodeMessage(data, model)
```

### Project Structure

- `model.go`: Core `Model` interface definition.
- `openai.go` / `anthropic.go`: Concrete implementations for each provider.
- `define.go`: Common logic for tools and function definitions.
- `template.go`: Parameter parsing templates based on reflection.
- `message.go`: Message interface and serialization tools.
- `response.go`: Response interface and statistics structures.

### License

Apache License 2.0
