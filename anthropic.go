package openllm

import (
	"context"
	"encoding/base64"
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
	var reasoning strings.Builder
	var tcalls []ToolCall
	var toolCallIndex int

	for _, block := range chatResp.Content {
		switch b := block.AsAny().(type) {
		case anthropic.TextBlock:
			content.WriteString(b.Text)
		case anthropic.ThinkingBlock:
			reasoning.WriteString(b.Thinking)
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
	answer := &llmmsg{
		role:      constants.RoleAssistant,
		content:   []ContentPart{{Type: constants.ContentPartTypeText, Text: content.String()}},
		reasoning: reasoning.String(),
		toolCalls: func() []*toolcall {
			if len(tcalls) == 0 {
				return nil
			}
			var gtc []*toolcall
			for _, tc := range tcalls {
				if tc, ok := tc.(*toolcall); ok {
					gtc = append(gtc, tc)
				}
			}
			return gtc
		}(),
	}

	usage := Usage{
		InputTokens:              int(chatResp.Usage.InputTokens),
		OutputTokens:             int(chatResp.Usage.OutputTokens),
		TotalTokens:              int(chatResp.Usage.InputTokens + chatResp.Usage.OutputTokens),
		CacheCreationInputTokens: int(chatResp.Usage.CacheCreationInputTokens),
		CacheReadInputTokens:     int(chatResp.Usage.CacheReadInputTokens),
	}
	duration := time.Since(start)
	meta := Meta{
		Provider:   constants.ProviderAnthropic,
		Model:      a.name,
		RequestID:  chatResp.ID,
		StopReason: string(chatResp.StopReason),
	}

	return &response{
		answer:   answer,
		tcalls:   tcalls,
		usage:    usage,
		duration: duration,
		meta:     meta,
	}, nil
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
		role      string
		content   strings.Builder
		reasoning strings.Builder
		callm     = make(map[int]*toolcall)
	)

	for stream.Next() {
		event := stream.Current()

		switch ev := event.AsAny().(type) {
		case anthropic.MessageStartEvent:
			if ev.Message.Role != "" {
				role = constants.RoleAssistant
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
			case anthropic.ThinkingDelta:
				if options.watcher != nil {
					if err := options.watcher.OnReasoning(d.Thinking); err != nil {
						return nil, err
					}
				}
				reasoning.WriteString(d.Thinking)
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

	answer := &llmmsg{
		role:      role,
		content:   []ContentPart{{Type: constants.ContentPartTypeText, Text: content.String()}},
		reasoning: reasoning.String(),
		toolCalls: func() []*toolcall {
			if len(tcalls) == 0 {
				return nil
			}
			var gtc []*toolcall
			for _, tc := range tcalls {
				if tc, ok := tc.(*toolcall); ok {
					gtc = append(gtc, tc)
				}
			}
			return gtc
		}(),
	}

	return &response{
		answer:   answer,
		tcalls:   tcalls,
		usage:    Usage{},
		duration: time.Since(start),
		meta: Meta{
			Provider: constants.ProviderAnthropic,
			Model:    a.name,
		},
	}, nil
}

// makeRequest builds an Anthropic MessageNewParams from ChatOptions and Message list.
// It converts messages to the Anthropic format, applies system prompt and temperature,
// and attaches tool definitions when provided.
func (a *anthropicLLM) makeRequest(opts *ChatOptions, messages []Message) (req anthropic.MessageNewParams, err error) {
	req.Model = anthropic.Model(a.name)
	req.MaxTokens = int64(4096) // Default max tokens

	// Set temperature (optional). If your SDK version requires param.Opt,
	// you can wire it here; otherwise omit to use server defaults.

	// Option: MaxTokens
	if opts.maxTokens != nil {
		req.MaxTokens = int64(*opts.maxTokens)
	}
	// Option: Temperature
	if opts.temperature != nil {
		req.Temperature = anthropic.Opt(*opts.temperature)
	}
	// Option: TopK
	if opts.topK != nil {
		req.TopK = anthropic.Opt(int64(*opts.topK))
	}
	// Option: TopP
	if opts.topP != nil {
		req.TopP = anthropic.Opt(*opts.topP)
	}

	// Option: ReasoningEffort
	if opts.reasoningEffort != nil {
		var budget int64
		switch *opts.reasoningEffort {
		case constants.ReasoningEffortLow:
			budget = 1024
		case constants.ReasoningEffortMedium:
			budget = 4096
		case constants.ReasoningEffortHigh:
			budget = 8192
		default:
			budget = 4096 // Default to Medium
		}

		// Ensure budget < max_tokens
		// If max_tokens is set, cap budget.
		// Note: Anthropic requires budget < max_tokens.
		// If max_tokens is not set in options, we used default 4096 above.
		maxTokens := req.MaxTokens
		if budget >= maxTokens {
			// Reserve some space for output?
			// Actually, Anthropic docs say: "budget_tokens must be less than max_tokens"
			// Let's cap it at maxTokens - 1 to be safe, or just reduce it.
			// If maxTokens is small (e.g. 1024), low budget (1024) would fail.
			if maxTokens > 64 {
				budget = maxTokens - 64 // Leave room for at least a small response
			} else {
				// Very small max_tokens, disable thinking or set to minimum?
				// Minimum budget is 1024 usually. If max_tokens < 1024, we can't enable thinking properly.
				// But let's just clamp to maxTokens-1 for API correctness attempt, though it might error.
				budget = maxTokens - 1
			}
		}

		if budget > 0 {
			req.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
		}
	}

	// Set system prompt
	if opts.prompt != "" {
		req.System = []anthropic.TextBlockParam{
			{Text: opts.prompt},
		}
	}

	// Convert messages
	var anthropicMessages []anthropic.MessageParam
	for _, message := range messages {
		msgParam, err := a.convertMessage(message)
		if err != nil {
			return req, err
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

// convertMessage transforms the unified Message (llmmsg) into Anthropic's MessageParam.
// It handles role mapping, content blocks, image conversion, and tool calls.
func (a *anthropicLLM) convertMessage(message Message) (anthropic.MessageParam, error) {
	// Cast to llmmsg to access internal structure
	msg, ok := message.(*llmmsg)
	if !ok {
		// Fallback for custom implementations (should ideally not happen with global factories)
		return anthropic.NewUserMessage(anthropic.NewTextBlock(message.Content())), nil
	}

	role := msg.role

	// Handle "tool" role (OpenAI) -> "user" role with ToolResultBlock (Anthropic)
	if role == constants.RoleTool {
		return anthropic.NewUserMessage(anthropic.NewToolResultBlock(
			msg.toolCallID,
			message.Content(),
			false, // isError
		)), nil
	}

	// Handle standard roles (user, assistant)
	var blocks []anthropic.ContentBlockParamUnion

	// 1. Process MultiContent (Images + Text) or standard Content
	if len(msg.content) > 0 {
		for _, part := range msg.content {
			switch part.Type {
			case constants.ContentPartTypeText:
				blocks = append(blocks, anthropic.NewTextBlock(part.Text))
			case constants.ContentPartTypeImageURL:
				if part.ImageURL == nil {
					continue
				}
				imgURL := part.ImageURL.URL

				// Image conversion logic (URL vs Base64)
				mediaType := "image/jpeg"
				data := imgURL
				isURL := false

				if strings.HasPrefix(imgURL, "http://") || strings.HasPrefix(imgURL, "https://") {
					isURL = true
				} else if idx := strings.Index(imgURL, ";base64,"); idx != -1 {
					prefix := imgURL[:idx]
					if strings.HasPrefix(prefix, "data:") {
						mediaType = strings.TrimPrefix(prefix, "data:")
					}
					data = imgURL[idx+len(";base64,"):]
				} else {
					// Magic number detection for raw base64
					if len(data) > 15 {
						prefixData := data
						if len(prefixData) > 64 {
							prefixData = prefixData[:64]
						}
						decoded, err := base64.StdEncoding.DecodeString(prefixData)
						if err == nil && len(decoded) > 4 {
							if len(decoded) >= 8 && string(decoded[0:8]) == "\x89PNG\r\n\x1a\n" {
								mediaType = "image/png"
							} else if len(decoded) >= 3 && string(decoded[0:3]) == "\xff\xd8\xff" {
								mediaType = "image/jpeg"
							} else if len(decoded) >= 6 && (string(decoded[0:6]) == "GIF87a" || string(decoded[0:6]) == "GIF89a") {
								mediaType = "image/gif"
							} else if len(decoded) >= 12 && string(decoded[0:4]) == "RIFF" && string(decoded[8:12]) == "WEBP" {
								mediaType = "image/webp"
							}
						}
					}
				}

				if isURL {
					blocks = append(blocks, anthropic.NewImageBlock(
						anthropic.URLImageSourceParam{
							URL: imgURL,
						},
					))
				} else {
					blocks = append(blocks, anthropic.NewImageBlockBase64(
						mediaType,
						data,
					))
				}
			}
		}
	}

	// 2. Process ToolCalls (Assistant role)
	if len(msg.toolCalls) > 0 {
		for _, tc := range msg.toolCalls {
			if tc.type_ != constants.ToolTypeFunction {
				continue
			}
			var input map[string]any
			if err := json.Unmarshal([]byte(tc.fcall.Arguments()), &input); err != nil {
				input = map[string]any{}
			}
			toolUse := anthropic.ToolUseBlockParam{
				ID:    tc.id,
				Name:  tc.fcall.Name(),
				Input: input,
			}
			blocks = append(blocks, anthropic.ContentBlockParamUnion{OfToolUse: &toolUse})
		}
	}

	// Construct final message based on role
	switch role {
	case constants.RoleUser:
		if len(blocks) == 0 {
			return anthropic.NewUserMessage(anthropic.NewTextBlock("")), nil
		}
		return anthropic.NewUserMessage(blocks...), nil
	case constants.RoleAssistant:
		if len(blocks) == 0 {
			return anthropic.NewAssistantMessage(anthropic.NewTextBlock("")), nil
		}
		return anthropic.NewAssistantMessage(blocks...), nil
	case constants.RoleSystem:
		// System messages should be handled separately (in req.System),
		// but if one slips here, treat as user or ignore?
		// Ideally makeRequest filters them out.
		return anthropic.NewUserMessage(anthropic.NewTextBlock(message.Content())), nil
	default:
		return anthropic.NewUserMessage(anthropic.NewTextBlock(message.Content())), nil
	}
}
