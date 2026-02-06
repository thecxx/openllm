package openllm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/thecxx/openllm/constants"
)

type anthropicLLM struct {
	name        string
	description string
	client      *anthropic.Client
}

// NewAnthropicLLM creates a new Model implementation for Anthropic's API.
func NewAnthropicLLM(name, description string, client *anthropic.Client) Model {
	return &anthropicLLM{name: name, description: description, client: client}
}

// NewAnthropicLLMWithAPIKey creates a new Model implementation with an API key.
func NewAnthropicLLMWithAPIKey(name, description, apiKey string) Model {
	client := anthropic.NewClient(option.WithAPIKey(apiKey))
	return &anthropicLLM{name: name, description: description, client: &client}
}

// Name returns the model identifier string.
func (a *anthropicLLM) Name() string {
	return a.name
}

// Description returns a human-readable description of the model.
func (a *anthropicLLM) Description() string {
	return a.description
}

// ChatCompletion performs a blocking chat completion request.
// It builds the request from messages and options, executes the call,
// and returns the final assistant message together with any tool-calls.
func (a *anthropicLLM) ChatCompletion(ctx context.Context, messages []Message, opts ...ChatOption) (resp Response, err error) {
	options := &ChatOptions{}
	// Set chat options
	for _, opt := range opts {
		opt(options)
	}

	req, err := a.makeRequest(options, messages)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	chatResp, err := a.client.Messages.New(ctx, req)
	if err != nil {
		return nil, err
	}

	// Defensive: ensure we have at least one content block
	if len(chatResp.Content) <= 0 {
		return nil, ErrEmptyChoices
	}

	var content strings.Builder
	var tcalls []ToolCall
	var toolCallIndex int

	for _, block := range chatResp.Content {
		switch b := block.AsAny().(type) {
		case anthropic.TextBlock:
			content.WriteString(b.Text)
		case anthropic.ToolUseBlock:
			argsJSON, err := json.Marshal(b.Input)
			if err != nil {
				return nil, err
			}
			tcalls = append(tcalls, &toolcall{
				index: toolCallIndex,
				id:    b.ID,
				type_: constants.ToolTypeFunction,
				fcall: funcall{
					name: b.Name,
					args: string(argsJSON),
				},
			})
			toolCallIndex++
		}
	}

	// Create anthropic message wrapper
	answer := &anthropicMsg{
		role:    string(chatResp.Role),
		content: content.String(),
		tcalls:  tcalls,
	}

	stats := Stats{
		Usage: Usage{
			InputTokens:              int(chatResp.Usage.InputTokens),
			OutputTokens:             int(chatResp.Usage.OutputTokens),
			TotalTokens:              int(chatResp.Usage.InputTokens + chatResp.Usage.OutputTokens),
			CacheCreationInputTokens: int(chatResp.Usage.CacheCreationInputTokens),
			CacheReadInputTokens:     int(chatResp.Usage.CacheReadInputTokens),
		},
		Duration: time.Since(start),
		Meta: Meta{
			Provider:   "anthropic",
			Model:      a.name,
			RequestID:  chatResp.ID,
			StopReason: string(chatResp.StopReason),
		},
	}
	return &response{answer: answer, tcalls: tcalls, stats: stats}, nil
}

// ChatCompletionStream performs a streaming chat completion request.
// It emits incremental content via the StreamEventHandler (if provided),
// collects streamed tool-call arguments, and returns the assembled answer
// and ordered tool-calls once the stream finishes.
func (a *anthropicLLM) ChatCompletionStream(ctx context.Context, messages []Message, opts ...ChatOption) (resp Response, err error) {
	options := &ChatOptions{}
	// Set chat options
	for _, opt := range opts {
		opt(options)
	}

	req, err := a.makeRequest(options, messages)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	stream := a.client.Messages.NewStreaming(ctx, req)

	var (
		role    string
		content strings.Builder
		callm   = make(map[int]*toolcall)
	)

	for stream.Next() {
		event := stream.Current()

		switch ev := event.AsAny().(type) {
		case anthropic.MessageStartEvent:
			if ev.Message.Role != "" {
				role = string(ev.Message.Role)
			}
		case anthropic.ContentBlockStartEvent:
			switch cb := ev.ContentBlock.AsAny().(type) {
			case anthropic.ToolUseBlock:
				tcall := &toolcall{
					index: int(ev.Index),
					id:    cb.ID,
					type_: constants.ToolTypeFunction,
					fcall: funcall{
						name: cb.Name,
					},
				}
				if options.watcher != nil {
					if err := options.watcher.OnToolCall(ctx, tcall, ""); err != nil {
						return nil, err
					}
				}
				callm[int(ev.Index)] = tcall
			}
		case anthropic.ContentBlockDeltaEvent:
			switch d := ev.Delta.AsAny().(type) {
			case anthropic.TextDelta:
				if options.watcher != nil {
					if err := options.watcher.OnContent(d.Text); err != nil {
						return nil, err
					}
				}
				content.WriteString(d.Text)
			case anthropic.InputJSONDelta:
				if tcall, found := callm[int(ev.Index)]; found {
					if options.watcher != nil {
						if err := options.watcher.OnToolCall(ctx, tcall, d.PartialJSON); err != nil {
							return nil, err
						}
					}
					tcall.fcall.writeArgs(d.PartialJSON)
				}
			}
		}
	}

	if err := stream.Err(); err != nil {
		if !errors.Is(err, io.EOF) {
			return nil, err
		}
	}

	if options.watcher != nil {
		if err := options.watcher.OnStop(); err != nil {
			return nil, err
		}
	}

	var tcalls = make([]ToolCall, 0)
	if len(callm) > 0 {
		for _, tcall := range callm {
			tcalls = append(tcalls, tcall)
		}
		sort.Slice(tcalls, func(i, j int) bool {
			return tcalls[i].Index() < tcalls[j].Index()
		})
	}

	answer := &anthropicMsg{
		role:    role,
		content: content.String(),
		tcalls:  tcalls,
	}

	stats := Stats{
		Usage:    Usage{},
		Duration: time.Since(start),
		Meta: Meta{
			Provider: "anthropic",
			Model:    a.name,
		},
	}
	return &response{answer: answer, tcalls: tcalls, stats: stats}, nil
}

// makeRequest builds an Anthropic MessageNewParams from ChatOptions and Message list.
// It converts messages to the Anthropic format, applies system prompt and temperature,
// and attaches tool definitions when provided.
func (a *anthropicLLM) makeRequest(opts *ChatOptions, messages []Message) (req anthropic.MessageNewParams, err error) {
	req.Model = anthropic.Model(a.name)
	// req.MaxTokens = int64(4096) // Default max tokens

	// Set temperature (optional). If your SDK version requires param.Opt,
	// you can wire it here; otherwise omit to use server defaults.

	// Set system prompt
	if opts.prompt != "" {
		req.System = []anthropic.TextBlockParam{
			{Text: opts.prompt},
		}
	}

	// Convert messages
	var anthropicMessages []anthropic.MessageParam
	for _, message := range messages {
		// Try to cast to anthropicMsg first for efficiency
		if msg, ok := message.(*anthropicMsg); ok {
			anthropicMessages = append(anthropicMessages, msg.toMessageParam())
			continue
		}

		// Try to cast to anthropicToolResultMsg
		if msg, ok := message.(*anthropicToolResultMsg); ok {
			anthropicMessages = append(anthropicMessages, msg.toMessageParam())
			continue
		}

		// Fallback: create message from role and content
		role := message.Role()
		content := message.Content()

		var msgParam anthropic.MessageParam
		switch role {
		case string(anthropic.MessageParamRoleUser):
			msgParam = anthropic.NewUserMessage(anthropic.NewTextBlock(content))
		case string(anthropic.MessageParamRoleAssistant):
			msgParam = anthropic.NewAssistantMessage(anthropic.NewTextBlock(content))
		default:
			// Default to user message
			msgParam = anthropic.NewUserMessage(anthropic.NewTextBlock(content))
		}
		anthropicMessages = append(anthropicMessages, msgParam)
	}

	req.Messages = anthropicMessages

	for _, tool := range opts.tools {
		var toolParam anthropic.ToolParam
		if def, ok := tool.Definition().(anthropic.ToolParam); ok {
			toolParam = def
		} else if def, ok := tool.Definition().(*FunctionDefinition); ok {
			// Convert generic FunctionDefinition to Anthropic ToolParam
			toolParam = anthropic.ToolParam{
				Name:        def.Name,
				Description: anthropic.String(def.Description),
				Strict:      anthropic.Bool(def.Strict),
			}

			// Handle InputSchema conversion from generic Parameters
			if schema, ok := def.Parameters.(anthropic.ToolInputSchemaParam); ok {
				toolParam.InputSchema = schema
			} else {
				// Default minimal valid schema
				toolParam.InputSchema = anthropic.ToolInputSchemaParam{
					Type:       "object",
					Properties: map[string]any{},
				}

				// Try JSON round-trip for complex structures or jsonschema.Definition
				if def.Parameters != nil {
					data, err := json.Marshal(def.Parameters)
					if err == nil {
						var inputSchema anthropic.ToolInputSchemaParam
						if err := json.Unmarshal(data, &inputSchema); err == nil && inputSchema.Type != "" {
							toolParam.InputSchema = inputSchema
						}
					}
				}
			}
		} else {
			// Try full JSON round-trip conversion for unknown types
			data, err := json.Marshal(tool.Definition())
			if err == nil {
				var tp anthropic.ToolParam
				if err := json.Unmarshal(data, &tp); err == nil {
					toolParam = tp
				}
			}
		}

		if toolParam.Name != "" {
			req.Tools = append(req.Tools, anthropic.ToolUnionParam{OfTool: &toolParam})
		}
	}

	return req, nil
}

// NewUserMessage creates a user-role message suitable for Anthropic.
func (a *anthropicLLM) NewUserMessage(content string, opts ...MessageOption) Message {
	var options MessageOptions
	for _, opt := range opts {
		opt(&options)
	}
	return &anthropicMsg{
		role:    string(anthropic.MessageParamRoleUser),
		content: content,
	}
}

// NewToolMessage creates a tool result message suitable for Anthropic.
func (a *anthropicLLM) NewToolMessage(tool ToolCall, result string) Message {
	return &anthropicToolResultMsg{
		toolUseID: tool.ID(),
		content:   result,
	}
}

// anthropicMsg implements Message interface using Anthropic's message format.
type anthropicMsg struct {
	role    string
	content string
	tcalls  []ToolCall
}

// Role implements Message.
func (m *anthropicMsg) Role() string {
	return m.role
}

// Content implements Message.
func (m *anthropicMsg) Content() string {
	return m.content
}

// toMessageParam converts anthropicMsg to Anthropic's MessageParam.
func (m *anthropicMsg) toMessageParam() anthropic.MessageParam {
	var blocks []anthropic.ContentBlockParamUnion
	if m.content != "" {
		blocks = append(blocks, anthropic.NewTextBlock(m.content))
	}
	for _, tc := range m.tcalls {
		var input map[string]interface{}
		if err := json.Unmarshal([]byte(tc.Function().Arguments()), &input); err != nil {
			input = map[string]interface{}{}
		}
		param := anthropic.ToolUseBlockParam{
			ID:    tc.ID(),
			Name:  tc.Function().Name(),
			Input: input,
		}
		blocks = append(blocks, anthropic.ContentBlockParamUnion{OfToolUse: &param})
	}

	switch m.role {
	case string(anthropic.MessageParamRoleUser):
		if len(blocks) == 0 {
			return anthropic.NewUserMessage(anthropic.NewTextBlock(""))
		}
		return anthropic.NewUserMessage(blocks...)
	case string(anthropic.MessageParamRoleAssistant):
		if len(blocks) == 0 {
			return anthropic.NewAssistantMessage(anthropic.NewTextBlock(""))
		}
		return anthropic.NewAssistantMessage(blocks...)
	default:
		return anthropic.NewUserMessage(anthropic.NewTextBlock(m.content))
	}
}

// anthropicToolResultMsg implements Message interface for tool results in Anthropic.
type anthropicToolResultMsg struct {
	toolUseID string
	content   string
}

// Role implements Message.
func (m *anthropicToolResultMsg) Role() string {
	return string(anthropic.MessageParamRoleUser)
}

// Content implements Message.
func (m *anthropicToolResultMsg) Content() string {
	return m.content
}

// toMessageParam converts anthropicToolResultMsg to Anthropic's MessageParam.
func (m *anthropicToolResultMsg) toMessageParam() anthropic.MessageParam {
	return anthropic.NewUserMessage(anthropic.NewToolResultBlock(m.toolUseID, m.content, false))
}
