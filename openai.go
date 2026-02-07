package openllm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sort"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"github.com/thecxx/openllm/constants"
)

type llm struct {
	name        string
	description string
	client      *openai.Client
}

// NewLLM creates a new Model implementation for a specific model name and client.
func NewLLM(name, description string, client *openai.Client) Model {
	return &llm{name: name, description: description, client: client}
}

// NewLLMWithAPIKey creates a new Model implementation with an auth token.
func NewLLMWithAPIKey(name, description, authToken string) Model {
	client := openai.NewClient(authToken)
	return &llm{name: name, description: description, client: client}
}

// Name returns the model identifier string.
func (l *llm) Name() string {
	return l.name
}

// Description returns a human-readable description of the model.
func (l *llm) Description() string {
	return l.description
}

// ChatCompletion performs a blocking chat completion request.
// It builds the request from messages and options, executes the call,
// and returns the final assistant message together with any tool-calls.
func (l *llm) ChatCompletion(ctx context.Context, messages []Message, opts ...ChatOption) (resp Response, err error) {
	options := &ChatOptions{}
	// Set chat options
	for _, opt := range opts {
		opt(options)
	}

	req, err := l.makeRequest(options, messages)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	chatResp, err := l.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, err
	}

	// Defensive: ensure we have at least one choice
	if len(chatResp.Choices) <= 0 {
		return nil, ErrEmptyChoices
	}

	choice := chatResp.Choices[0]
	var tcalls []ToolCall
	if n := len(choice.Message.ToolCalls); n > 0 {
		tcalls = make([]ToolCall, 0, n)
	}

	for _, call := range choice.Message.ToolCalls {
		if call.Index == nil {
			continue
		}
		index := copyInt(*call.Index)
		if call.Type == openai.ToolTypeFunction && call.Function.Name != "" {
			tcalls = append(tcalls, &toolcall{
				index: index,
				id:    call.ID,
				type_: constants.ToolTypeFunction,
				fcall: funcall{
					name: call.Function.Name,
					args: call.Function.Arguments,
				},
			})
		}
	}

	usage := Usage{
		InputTokens:  chatResp.Usage.PromptTokens,
		OutputTokens: chatResp.Usage.CompletionTokens,
		TotalTokens:  chatResp.Usage.TotalTokens,
	}
	if chatResp.Usage.PromptTokensDetails != nil {
		usage.CachedTokens = chatResp.Usage.PromptTokensDetails.CachedTokens
	}
	if chatResp.Usage.CompletionTokensDetails != nil {
		usage.ReasoningTokens = chatResp.Usage.CompletionTokensDetails.ReasoningTokens
	}

	meta := Meta{
		Provider:          constants.ProviderOpenAI,
		Model:             chatResp.Model,
		RequestID:         chatResp.ID,
		SystemFingerprint: chatResp.SystemFingerprint,
		StopReason:        string(choice.FinishReason),
	}
	duration := time.Since(start)

	return &response{
		answer: &llmmsg{
			role:      choice.Message.Role,
			reasoning: choice.Message.ReasoningContent,
			refusal:   choice.Message.Refusal,
			content: func() []ContentPart {
				if choice.Message.Content != "" {
					return []ContentPart{{Type: constants.ContentPartTypeText, Text: choice.Message.Content}}
				}
				var parts []ContentPart
				for _, p := range choice.Message.MultiContent {
					if p.Type == openai.ChatMessagePartTypeText {
						parts = append(parts, ContentPart{Type: constants.ContentPartTypeText, Text: p.Text})
					} else if p.Type == openai.ChatMessagePartTypeImageURL && p.ImageURL != nil {
						parts = append(parts, ContentPart{
							Type:     constants.ContentPartTypeImageURL,
							ImageURL: &ImageURL{URL: p.ImageURL.URL, Detail: string(p.ImageURL.Detail)},
						})
					}
				}
				return parts
			}(),
			toolcalls: func() []*toolcall {
				if len(tcalls) == 0 {
					return nil
				}
				var gtc []*toolcall
				for _, tcall := range tcalls {
					gtc = append(gtc, &toolcall{
						index: tcall.Index(),
						id:    tcall.ID(),
						type_: tcall.Type(),
						fcall: funcall{
							name: tcall.Function().Name(),
							args: tcall.Function().Arguments(),
						},
					})
				}
				return gtc
			}(),
		},
		tcalls:   tcalls,
		usage:    usage,
		meta:     meta,
		duration: duration,
	}, nil
}

// ChatCompletionStream performs a streaming chat completion request.
// It emits incremental content via the StreamEventHandler (if provided),
// collects streamed tool-call arguments, and returns the assembled answer
// and ordered tool-calls once the stream finishes.
func (l *llm) ChatCompletionStream(ctx context.Context, messages []Message, opts ...ChatOption) (resp Response, err error) {
	options := &ChatOptions{}
	// Set chat options
	for _, opt := range opts {
		opt(options)
	}

	req, err := l.makeRequest(options, messages)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	stream, err := l.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	var (
		role      string
		content   strings.Builder
		reasoning strings.Builder
		refusal   strings.Builder
		rawmsg    openai.ChatCompletionMessage
		callm     = make(map[int]*toolcall)
	)

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		resp, err := stream.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}

		// Ignore empty payloads defensively
		if len(resp.Choices) <= 0 {
			continue
		}
		choice := resp.Choices[0]

		// Set role
		if choice.Delta.Role != "" && role == "" {
			role = choice.Delta.Role
		}

		if choice.Delta.ReasoningContent != "" {
			if options.watcher != nil {
				if err = options.watcher.OnReasoning(choice.Delta.ReasoningContent); err != nil {
					return nil, err
				}
			}
			reasoning.WriteString(choice.Delta.ReasoningContent)
		}

		if choice.Delta.Content != "" {
			if options.watcher != nil {
				if err = options.watcher.OnContent(choice.Delta.Content); err != nil {
					return nil, err
				}
			}
			content.WriteString(choice.Delta.Content)
		}

		if choice.Delta.Refusal != "" {
			if options.watcher != nil {
				if err = options.watcher.OnRefusal(choice.Delta.Refusal); err != nil {
					return nil, err
				}
			}
			refusal.WriteString(choice.Delta.Refusal)
		}

		if len(choice.Delta.ToolCalls) > 0 {
			for _, call := range choice.Delta.ToolCalls {
				if call.Index == nil {
					continue
				}
				index := copyInt(*call.Index)
				if call.Type == openai.ToolTypeFunction && call.Function.Name != "" {
					tcall := &toolcall{
						index: index,
						id:    call.ID,
						type_: constants.ToolTypeFunction,
						fcall: funcall{
							name: call.Function.Name,
						},
					}
					if options.watcher != nil {
						if err = options.watcher.OnToolCall(ctx, tcall, ""); err != nil {
							return nil, err
						}
					}
					callm[index] = tcall
				}

				if call.Function.Arguments != "" {
					tcall, found := callm[index]
					if found {
						if options.watcher != nil {
							if err = options.watcher.OnToolCall(ctx, tcall, call.Function.Arguments); err != nil {
								return nil, err
							}
						}
						tcall.fcall.writeArgs(call.Function.Arguments)
					}
				}
			}
		}
	}

	if options.watcher != nil {
		if err := options.watcher.OnStop(); err != nil {
			return nil, err
		}
	}

	rawmsg.Role = role
	rawmsg.Content = content.String()
	rawmsg.ReasoningContent = reasoning.String()
	rawmsg.Refusal = refusal.String()

	var tcalls = make([]ToolCall, 0)
	if len(callm) > 0 {
		for _, tcall := range callm {
			tcalls = append(tcalls, tcall)
		}
		sort.Slice(tcalls, func(i, j int) bool {
			return tcalls[i].Index() < tcalls[j].Index()
		})
		for _, tcall := range tcalls {
			index := tcall.Index()
			rawmsg.ToolCalls = append(rawmsg.ToolCalls, openai.ToolCall{
				Index: &index,
				ID:    tcall.ID(),
				Type:  openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      tcall.Function().Name(),
					Arguments: tcall.Function().Arguments(),
				},
			})
		}
	}

	return &response{
		answer: &llmmsg{
			role: rawmsg.Role,
			content: func() []ContentPart {
				if rawmsg.Content != "" {
					return []ContentPart{{Type: constants.ContentPartTypeText, Text: rawmsg.Content}}
				}
				var parts []ContentPart
				for _, p := range rawmsg.MultiContent {
					if p.Type == openai.ChatMessagePartTypeText {
						parts = append(parts, ContentPart{Type: constants.ContentPartTypeText, Text: p.Text})
					} else if p.Type == openai.ChatMessagePartTypeImageURL && p.ImageURL != nil {
						parts = append(parts, ContentPart{
							Type:     constants.ContentPartTypeImageURL,
							ImageURL: &ImageURL{URL: p.ImageURL.URL, Detail: string(p.ImageURL.Detail)},
						})
					}
				}
				return parts
			}(),
			reasoning: rawmsg.ReasoningContent,
			refusal:   rawmsg.Refusal,
			toolcalls: func() []*toolcall {
				if len(tcalls) == 0 {
					return nil
				}
				var gtc []*toolcall
				for _, tcall := range tcalls {
					gtc = append(gtc, &toolcall{
						index: tcall.Index(),
						id:    tcall.ID(),
						type_: tcall.Type(),
						fcall: funcall{
							name: tcall.Function().Name(),
							args: tcall.Function().Arguments(),
						},
					})
				}
				return gtc
			}(),
		},
		tcalls:   tcalls,
		usage:    Usage{},
		duration: time.Since(start),
		meta: Meta{
			Provider: constants.ProviderOpenAI,
			Model:    l.name,
		},
	}, nil
}

// makeRequest builds an OpenAI ChatCompletionRequest from ChatOptions and Message list.
// It converts messages to the OpenAI format, applies system prompt and temperature,
// and attaches tool definitions when provided.
func (l *llm) makeRequest(opts *ChatOptions, messages []Message) (req openai.ChatCompletionRequest, err error) {
	req.Model = l.name
	// Option: MaxTokens
	if opts.maxTokens != nil {
		req.MaxCompletionTokens = *opts.maxTokens
		// req.MaxTokens = *opts.maxTokens
	}
	// Option: Temperature
	if opts.temperature != nil {
		req.Temperature = float32(*opts.temperature)
	}
	// Option: TopP
	if opts.topP != nil {
		req.TopP = float32(*opts.topP)
	}

	// Option: ReasoningEffort
	if opts.reasoningEffort != nil {
		switch *opts.reasoningEffort {
		case constants.ReasoningEffortLow, constants.ReasoningEffortMedium, constants.ReasoningEffortHigh:
			req.ReasoningEffort = *opts.reasoningEffort
		default:
			// Fallback or ignore invalid values
			req.ReasoningEffort = constants.ReasoningEffortMedium
		}
	}

	if opts.prompt != "" {
		req.Messages = append(req.Messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: opts.prompt,
		})
	}

	for _, message := range messages {
		openaiMsg, err := l.convertMessage(message)
		if err != nil {
			// Fallback? Or return error?
			// Since convertMessage returns nil error for fallback currently,
			// this should be fine.
			// But for safety:
			return req, err
		}
		req.Messages = append(req.Messages, openaiMsg)
	}

	for _, tool := range opts.tools {
		var fn *openai.FunctionDefinition
		if def, ok := tool.Definition().(*openai.FunctionDefinition); ok {
			fn = def
		} else if def, ok := tool.Definition().(*function); ok {
			fn = &openai.FunctionDefinition{
				Name:        def.name,
				Description: def.description,
				Parameters:  def.parameters,
				Strict:      def.strict,
			}
		} else {
			// Try JSON round-trip conversion for compatibility
			data, err := json.Marshal(tool.Definition())
			if err == nil {
				var def openai.FunctionDefinition
				if err := json.Unmarshal(data, &def); err == nil {
					fn = &def
				}
			}
		}

		if fn != nil {
			req.Tools = append(req.Tools, openai.Tool{
				Type:     openai.ToolType(tool.Type()),
				Function: fn,
			})
		}
	}

	return req, nil
}

// convertMessage transforms the unified Message (llmmsg) into OpenAI's ChatCompletionMessage.
func (l *llm) convertMessage(message Message) (openai.ChatCompletionMessage, error) {
	// Cast to llmmsg to access internal structure
	msg, ok := message.(*llmmsg)
	if !ok {
		// Fallback for custom implementations
		return openai.ChatCompletionMessage{
			Role:    message.Role(),
			Content: message.Content(),
		}, nil
	}

	raw := openai.ChatCompletionMessage{
		Role:             msg.role,
		ReasoningContent: msg.reasoning,
		ToolCallID:       msg.toolcallID,
	}

	// Handle Content (Text + Images)
	if len(msg.content) > 0 {
		// If simple text (length 1 and type text), can use Content field,
		// but MultiContent is more robust for general cases.
		// However, standard OpenAI client might prefer Content field for simple text.
		// Let's check if it's pure text.
		isPureText := true
		for _, part := range msg.content {
			if part.Type != constants.ContentPartTypeText {
				isPureText = false
				break
			}
		}

		if isPureText && len(msg.content) == 1 {
			raw.Content = msg.content[0].Text
		} else {
			for _, part := range msg.content {
				switch part.Type {
				case constants.ContentPartTypeText:
					raw.MultiContent = append(raw.MultiContent, openai.ChatMessagePart{
						Type: openai.ChatMessagePartTypeText,
						Text: part.Text,
					})
				case constants.ContentPartTypeImageURL:
					if part.ImageURL != nil {
						raw.MultiContent = append(raw.MultiContent, openai.ChatMessagePart{
							Type: openai.ChatMessagePartTypeImageURL,
							ImageURL: &openai.ChatMessageImageURL{
								URL:    part.ImageURL.URL,
								Detail: openai.ImageURLDetail(part.ImageURL.Detail),
							},
						})
					}
				}
			}
		}
	}

	// Handle ToolCalls
	if len(msg.toolcalls) > 0 {
		for _, tcall := range msg.toolcalls {
			index := tcall.index
			raw.ToolCalls = append(raw.ToolCalls, openai.ToolCall{
				Index: &index,
				ID:    tcall.id,
				Type:  openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      tcall.fcall.Name(),
					Arguments: tcall.fcall.Arguments(),
				},
			})
		}
	}

	return raw, nil
}

// copyInt returns a value copy of the provided int.
// It exists mainly to document the intent when copying pointer-based indices.
func copyInt(i int) int { return i }
