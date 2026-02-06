package openllm

import (
	"encoding/json"

	openai "github.com/sashabaranov/go-openai"
	"github.com/thecxx/openllm/constants"
)

// MessageOptions collects per-message configuration such as image URLs
// for multi-modal user messages.
type MessageOptions struct {
	// imageURLs is the set of image parts to attach to a user message.
	imageURLs []ImageURL
}

// ImageURL represents an image URL with detail level for multi-modal messages.
type ImageURL struct {
	URL    string
	Detail string
}

// MessageOption applies a configuration to MessageOptions.
// Multiple options can be combined; they are applied in the order provided.
type MessageOption func(opts *MessageOptions)

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

// Message represents a minimal conversational unit.
// It exposes only the role and textual content.
type Message interface {
	// Role returns the logical role of the message
	// (e.g. system, assistant, user).
	Role() string

	// Content returns the textual content of the message.
	Content() string
}

// WireMessage is the agnostic, serializable format used for message persistence.
// It normalizes provider-specific structures (OpenAI/Anthropic) into a single format
// that can be safely stored and later reconstructed for any supported model.
type WireMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []WireToolCall `json:"tool_calls,omitempty"`
	Images     []ImageURL     `json:"images,omitempty"`
}

// WireToolCall represents a tool invocation in the serializable WireMessage format.
type WireToolCall struct {
	Index int    `json:"index"`
	ID    string `json:"id"`
	Type  string `json:"type"`
	Name  string `json:"name"`
	Args  string `json:"args"`
}

type baseMessage struct {
	role    string
	content string
}

func (m *baseMessage) Role() string    { return m.role }
func (m *baseMessage) Content() string { return m.content }

// EncodeMessage serializes a Message into a JSON-encoded byte slice.
// It handles provider-specific message structures (like OpenAI's tool calls or Anthropic's images)
// and normalizes them into a unified WireMessage format for persistence.
func EncodeMessage(msg Message) ([]byte, error) {
	wm := WireMessage{Role: msg.Role(), Content: msg.Content()}
	switch t := msg.(type) {
	case *llmmsg:
		// If it's an OpenAI tool result message, ensure role is "tool" for cross-provider compatibility
		if t.rawmsg.ToolCallID != "" && t.rawmsg.Role == openai.ChatMessageRoleTool {
			wm.Role = constants.RoleTool
			wm.ToolCallID = t.rawmsg.ToolCallID
		}
		for _, part := range t.rawmsg.MultiContent {
			if part.Type == openai.ChatMessagePartTypeImageURL && part.ImageURL != nil {
				wm.Images = append(wm.Images, ImageURL{URL: part.ImageURL.URL, Detail: string(part.ImageURL.Detail)})
			}
		}
		for _, tc := range t.rawmsg.ToolCalls {
			wm.ToolCalls = append(wm.ToolCalls, WireToolCall{
				Index: copyInt(*tc.Index), // openai tool call index is pointer
				ID:    tc.ID,
				Type:  string(tc.Type),
				Name:  tc.Function.Name,
				Args:  tc.Function.Arguments,
			})
		}
	case *anthropicMsg:
		for _, tc := range t.tcalls {
			wm.ToolCalls = append(wm.ToolCalls, WireToolCall{
				Index: tc.Index(),
				ID:    tc.ID(),
				Type:  tc.Type(),
				Name:  tc.Function().Name(),
				Args:  tc.Function().Arguments(),
			})
		}
	case *anthropicToolResultMsg:
		// Mark as "tool" role for Anthropic tool results to unify storage format
		wm.Role = constants.RoleTool
		wm.ToolCallID = t.toolUseID
	}
	return json.Marshal(wm)
}

// DecodeMessage deserializes a JSON-encoded byte slice back into a Message object.
// The resulting Message is optimized for the provided Model type, ensuring that
// tool calls and metadata are correctly reconstructed for the target provider.
func DecodeMessage(model Model, data []byte) (Message, error) {
	var wm WireMessage
	if err := json.Unmarshal(data, &wm); err != nil {
		return nil, err
	}

	// Handle tool result messages specially
	// We convert the unified "tool" role back to the provider's expected role
	if wm.Role == constants.RoleTool && wm.ToolCallID != "" {
		switch model.(type) {
		case *llm:
			// OpenAI uses "tool" role for tool results
			raw := openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				ToolCallID: wm.ToolCallID,
				Content:    wm.Content,
			}
			return &llmmsg{rawmsg: raw}, nil
		case *anthropicLLM:
			// Anthropic treats tool results as a special kind of "user" message content.
			// anthropicToolResultMsg.Role() will return constants.RoleUser.
			return &anthropicToolResultMsg{
				toolUseID: wm.ToolCallID,
				content:   wm.Content,
			}, nil
		default:
			return &baseMessage{role: wm.Role, content: wm.Content}, nil
		}
	}

	// Handle standard messages (user, assistant, system) and assistant tool calls
	switch model.(type) {
	case *llm:
		raw := openai.ChatCompletionMessage{
			Role: wm.Role,
		}
		if len(wm.Images) > 0 {
			for _, img := range wm.Images {
				raw.MultiContent = append(raw.MultiContent, openai.ChatMessagePart{
					Type:     openai.ChatMessagePartTypeImageURL,
					ImageURL: &openai.ChatMessageImageURL{URL: img.URL, Detail: openai.ImageURLDetail(img.Detail)},
				})
			}
			raw.MultiContent = append(raw.MultiContent, openai.ChatMessagePart{Type: openai.ChatMessagePartTypeText, Text: wm.Content})
		} else {
			raw.Content = wm.Content
		}
		if len(wm.ToolCalls) > 0 {
			for _, tc := range wm.ToolCalls {
				idx := tc.Index
				raw.ToolCalls = append(raw.ToolCalls, openai.ToolCall{
					Index: &idx,
					ID:    tc.ID,
					Type:  openai.ToolType(tc.Type),
					Function: openai.FunctionCall{
						Name:      tc.Name,
						Arguments: tc.Args,
					},
				})
			}
		}
		return &llmmsg{rawmsg: raw}, nil
	case *anthropicLLM:
		var tcalls []ToolCall
		for _, tc := range wm.ToolCalls {
			tcalls = append(tcalls, &toolcall{
				index: tc.Index,
				id:    tc.ID,
				type_: tc.Type,
				fcall: funcall{
					name: tc.Name,
					args: tc.Args,
				},
			})
		}
		return &anthropicMsg{role: wm.Role, content: wm.Content, tcalls: tcalls}, nil
	default:
		return &baseMessage{role: wm.Role, content: wm.Content}, nil
	}
}
