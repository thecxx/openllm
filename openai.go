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

	stats := Stats{
		Usage: Usage{
			InputTokens:  chatResp.Usage.PromptTokens,
			OutputTokens: chatResp.Usage.CompletionTokens,
			TotalTokens:  chatResp.Usage.TotalTokens,
		},
		Duration: time.Since(start),
		Meta: Meta{
			Provider:          constants.ProviderOpenAI,
			Model:             chatResp.Model,
			RequestID:         chatResp.ID,
			SystemFingerprint: chatResp.SystemFingerprint,
			StopReason:        string(choice.FinishReason),
		},
	}
	if chatResp.Usage.PromptTokensDetails != nil {
		stats.Usage.CachedTokens = chatResp.Usage.PromptTokensDetails.CachedTokens
	}
	if chatResp.Usage.CompletionTokensDetails != nil {
		stats.Usage.ReasoningTokens = chatResp.Usage.CompletionTokensDetails.ReasoningTokens
	}
	return &response{answer: &llmmsg{rawmsg: choice.Message}, tcalls: tcalls, stats: stats}, nil
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
		role    string
		content strings.Builder
		rawmsg  openai.ChatCompletionMessage
		callm   = make(map[int]*toolcall)
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

		if len(choice.Delta.ToolCalls) <= 0 {
			if choice.Delta.Content != "" {
				if options.watcher != nil {
					if err = options.watcher.OnContent(choice.Delta.Content); err != nil {
						return nil, err
					}
				}
				content.WriteString(choice.Delta.Content)
			}
		} else {
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
							args: call.Function.Arguments,
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

	stats := Stats{
		Usage:    Usage{},
		Duration: time.Since(start),
		Meta: Meta{
			Provider: constants.ProviderOpenAI,
			Model:    l.name,
		},
	}
	return &response{answer: &llmmsg{rawmsg: rawmsg}, tcalls: tcalls, stats: stats}, nil
}

// makeRequest builds an OpenAI ChatCompletionRequest from ChatOptions and Message list.
// It converts messages to the OpenAI format, applies system prompt and temperature,
// and attaches tool definitions when provided.
func (l *llm) makeRequest(opts *ChatOptions, messages []Message) (req openai.ChatCompletionRequest, err error) {
	req.Model = l.name
	// Set temperature
	if opts.temperature != nil {
		req.Temperature = *opts.temperature
	}

	if opts.prompt != "" {
		req.Messages = append(req.Messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: opts.prompt,
		})
	}

	for _, message := range messages {
		// Try to cast to llmmsg first for efficiency
		if msg, ok := message.(*llmmsg); ok {
			req.Messages = append(req.Messages, msg.rawmsg)
			continue
		}

		// Fallback to JSON round-trip or basic fields
		// Since we don't know the implementation, we can only rely on Role and Content if it's not a BaseMessage
		// Or try to marshal it if it supports JSON
		data, err := json.Marshal(message)
		if err == nil {
			var openaiMsg openai.ChatCompletionMessage
			if err := json.Unmarshal(data, &openaiMsg); err == nil {
				req.Messages = append(req.Messages, openaiMsg)
				continue
			}
		}

		// Last resort: simple text message
		req.Messages = append(req.Messages, openai.ChatCompletionMessage{
			Role:    message.Role(),
			Content: message.Content(),
		})
	}

	for _, tool := range opts.tools {
		var fn *openai.FunctionDefinition
		if def, ok := tool.Definition().(*openai.FunctionDefinition); ok {
			fn = def
		} else if def, ok := tool.Definition().(*FunctionDefinition); ok {
			fn = &openai.FunctionDefinition{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  def.Parameters,
				Strict:      def.Strict,
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

// copyInt returns a value copy of the provided int.
// It exists mainly to document the intent when copying pointer-based indices.
func copyInt(i int) int { return i }

// NewUserMessage creates a user-role message suitable for OpenAI.
func (l *llm) NewUserMessage(content string, opts ...MessageOption) Message {
	var options MessageOptions
	for _, opt := range opts {
		opt(&options)
	}
	rawmsg := openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser}
	if len(options.imageURLs) <= 0 {
		rawmsg.Content = content
	} else {
		for _, imageURL := range options.imageURLs {
			rawmsg.MultiContent = append(rawmsg.MultiContent, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL:    imageURL.URL,
					Detail: openai.ImageURLDetail(imageURL.Detail),
				},
			})
		}
		rawmsg.MultiContent = append(rawmsg.MultiContent, openai.ChatMessagePart{
			Type: openai.ChatMessagePartTypeText,
			Text: content,
		})
	}
	return &llmmsg{rawmsg: rawmsg}
}

// NewToolMessage creates a tool result message suitable for OpenAI.
func (l *llm) NewToolMessage(tool ToolCall, result string) Message {
	return &llmmsg{
		rawmsg: openai.ChatCompletionMessage{
			Role:       openai.ChatMessageRoleTool,
			ToolCallID: tool.ID(),
			Content:    result,
		},
	}
}

// llmmsg implements Message interface using OpenAI's message format internally.
type llmmsg struct {
	rawmsg openai.ChatCompletionMessage
}

// Role implements Message.
func (m *llmmsg) Role() string {
	return m.rawmsg.Role
}

// Content implements Message.
func (m *llmmsg) Content() string {
	if m.rawmsg.Content != "" {
		return m.rawmsg.Content
	}
	for _, content := range m.rawmsg.MultiContent {
		if content.Type == openai.ChatMessagePartTypeText {
			return content.Text
		}
	}
	return ""
}

// MarshalJSON implements json.Marshaler.
func (m *llmmsg) MarshalJSON() ([]byte, error) {
	return json.Marshal(m.rawmsg)
}

// UnmarshalJSON implements json.Unmarshaler.
func (m *llmmsg) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, &m.rawmsg)
}

// WithImageURL adds an image URL with automatic detail selection for OpenAI.
func WithImageURL(imageURL string) MessageOption {
	return func(opts *MessageOptions) {
		opts.imageURLs = append(opts.imageURLs, ImageURL{
			URL:    imageURL,
			Detail: constants.ImageURLDetailAuto,
		})
	}
}

// WithImageURLDetail adds an image URL with an explicit detail level for OpenAI.
func WithImageURLDetail(imageURL string, detail string) MessageOption {
	if detail != constants.ImageURLDetailHigh &&
		detail != constants.ImageURLDetailLow &&
		detail != constants.ImageURLDetailAuto {
		detail = constants.ImageURLDetailAuto
	}
	return func(opts *MessageOptions) {
		opts.imageURLs = append(opts.imageURLs, ImageURL{
			URL:    imageURL,
			Detail: detail,
		})
	}
}
