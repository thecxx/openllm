package openllm

import (
	"encoding/json"
	"strings"

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
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// MessageOption applies a configuration to MessageOptions.
// Multiple options can be combined; they are applied in the order provided.
type MessageOption func(opts *MessageOptions)

// WithImageURL adds an image URL with automatic detail selection for OpenAI.
func WithImageURL(imageURL string) MessageOption {
	return WithImageURLDetail(imageURL, constants.ImageURLDetailAuto)
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

	// Reasoning returns the reasoning/thinking content of the message (if any).
	Reasoning() string
}

// NewUserMessage creates a user-role message suitable for any model.
func NewUserMessage(content string, opts ...MessageOption) Message {
	var options MessageOptions
	for _, opt := range opts {
		opt(&options)
	}
	msg := &llmmsg{
		role: constants.RoleUser,
	}

	if len(options.imageURLs) == 0 {
		msg.content = []ContentPart{
			{Type: constants.ContentPartTypeText, Text: content},
		}
	} else {
		// Mixed content: Text + Images
		for _, img := range options.imageURLs {
			msg.content = append(msg.content, ContentPart{
				Type:     constants.ContentPartTypeImageURL,
				ImageURL: &img,
			})
		}
		if content != "" {
			msg.content = append(msg.content, ContentPart{
				Type: constants.ContentPartTypeText,
				Text: content,
			})
		}
	}
	return msg
}

// NewToolMessage creates a tool result message suitable for any model.
func NewToolMessage(tool ToolCall, result string) Message {
	return &llmmsg{
		role:       constants.RoleTool,
		toolCallID: tool.ID(),
		content: []ContentPart{
			{Type: constants.ContentPartTypeText, Text: result},
		},
	}
}

// NewSystemMessage creates a system-role message suitable for any model.
func NewSystemMessage(content string) Message {
	return &llmmsg{
		role: constants.RoleSystem,
		content: []ContentPart{
			{Type: constants.ContentPartTypeText, Text: content},
		},
	}
}

// NewAssistantMessage creates an assistant-role message suitable for any model.
func NewAssistantMessage(content string, toolCalls ...ToolCall) Message {
	msg := &llmmsg{
		role: constants.RoleAssistant,
	}
	if content != "" {
		msg.content = []ContentPart{
			{Type: constants.ContentPartTypeText, Text: content},
		}
	}
	if len(toolCalls) > 0 {
		for _, tc := range toolCalls {
			msg.toolCalls = append(msg.toolCalls, &toolcall{
				index: tc.Index(),
				id:    tc.ID(),
				type_: tc.Type(),
				fcall: funcall{
					name: tc.Function().Name(),
					args: tc.Function().Arguments(),
				},
			})
		}
	}
	return msg
}

// ContentPart represents a part of a multi-modal message.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// llmmsg implements Message interface using a unified structure.
type llmmsg struct {
	role       string
	content    []ContentPart
	toolCalls  []*toolcall
	toolCallID string
	reasoning  string
	refusal    string
	name       string
}

// Role implements Message.
func (m *llmmsg) Role() string {
	return m.role
}

// Content implements Message.
func (m *llmmsg) Content() string {
	var sb strings.Builder
	for _, part := range m.content {
		if part.Type == constants.ContentPartTypeText {
			sb.WriteString(part.Text)
		}
	}
	return sb.String()
}

// Reasoning implements Message.
func (m *llmmsg) Reasoning() string {
	return m.reasoning
}

// MarshalJSON implements json.Marshaler.
func (m *llmmsg) MarshalJSON() ([]byte, error) {
	// We'll use a structure compatible with our previous WireMessage but cleaner.
	type alias struct {
		Role       string        `json:"role"`
		Content    []ContentPart `json:"content,omitempty"`
		ToolCalls  []*toolcall   `json:"tool_calls,omitempty"`
		ToolCallID string        `json:"tool_call_id,omitempty"`
		Reasoning  string        `json:"reasoning,omitempty"`
		Refusal    string        `json:"refusal,omitempty"`
		Name       string        `json:"name,omitempty"`
	}
	return json.Marshal(&alias{
		Role:       m.role,
		Content:    m.content,
		ToolCalls:  m.toolCalls,
		ToolCallID: m.toolCallID,
		Reasoning:  m.reasoning,
		Refusal:    m.refusal,
		Name:       m.name,
	})
}

// UnmarshalJSON implements json.Unmarshaler.
func (m *llmmsg) UnmarshalJSON(data []byte) error {
	type alias struct {
		Role       string        `json:"role"`
		Content    []ContentPart `json:"content,omitempty"`
		ToolCalls  []*toolcall   `json:"tool_calls,omitempty"`
		ToolCallID string        `json:"tool_call_id,omitempty"`
		Reasoning  string        `json:"reasoning,omitempty"`
		Refusal    string        `json:"refusal,omitempty"`
		Name       string        `json:"name,omitempty"`
	}
	var tmp alias
	if err := json.Unmarshal(data, &tmp); err != nil {
		return err
	}
	m.role = tmp.Role
	m.content = tmp.Content
	m.toolCalls = tmp.ToolCalls
	m.toolCallID = tmp.ToolCallID
	m.reasoning = tmp.Reasoning
	m.refusal = tmp.Refusal
	m.name = tmp.Name
	return nil
}

// EncodeMessage serializes a Message into a JSON-encoded byte slice.
func EncodeMessage(msg Message) ([]byte, error) {
	if m, ok := msg.(json.Marshaler); ok {
		return m.MarshalJSON()
	}
	return json.Marshal(msg)
}

// DecodeMessage deserializes a JSON-encoded byte slice back into a Message object.
func DecodeMessage(data []byte) (Message, error) {
	var m llmmsg
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}
