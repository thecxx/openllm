package openllm

import (
	"encoding/json"

	openai "github.com/sashabaranov/go-openai"
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

// Message represents a minimal conversational unit.
// It exposes only the role and textual content.
type Message interface {
	// Role returns the logical role of the message
	// (e.g. system, assistant, user).
	Role() string

	// Content returns the textual content of the message.
	Content() string
}

type WireMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []WireToolCall `json:"tool_calls,omitempty"`
	Images     []ImageURL     `json:"images,omitempty"`
}

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

func EncodeMessage(msg Message) ([]byte, error) {
	wm := WireMessage{Role: msg.Role(), Content: msg.Content()}
	switch t := msg.(type) {
	case *llmmsg:
		if t.rawmsg.ToolCallID != "" && t.rawmsg.Role == openai.ChatMessageRoleTool {
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
		wm.ToolCallID = t.toolUseID
	}
	return json.Marshal(wm)
}

func DecodeMessage(data []byte, model Model) (Message, error) {
	var wm WireMessage
	if err := json.Unmarshal(data, &wm); err != nil {
		return nil, err
	}
	if wm.Role == "tool" && wm.ToolCallID != "" {
		switch model.(type) {
		case *llm:
			raw := openai.ChatCompletionMessage{Role: openai.ChatMessageRoleTool, ToolCallID: wm.ToolCallID, Content: wm.Content}
			return &llmmsg{rawmsg: raw}, nil
		case *anthropicLLM:
			return &anthropicToolResultMsg{toolUseID: wm.ToolCallID, content: wm.Content}, nil
		default:
			return &baseMessage{role: wm.Role, content: wm.Content}, nil
		}
	}
	switch model.(type) {
	case *llm:
		raw := openai.ChatCompletionMessage{}
		raw.Role = wm.Role
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
		raw.Role = wm.Role
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
