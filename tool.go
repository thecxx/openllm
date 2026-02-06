package openllm

import (
	"strings"
)

// Tool describes a callable capability the model may invoke during generation.
// For OpenAI function calling, Type should be "function" and Definition should
// be an *openai.FunctionDefinition.
type Tool interface {
	// Type returns the category or kind of the tool.
	// For OpenAI function calling this should be "function".
	Type() string

	// Definition returns the configuration/metadata of the tool.
	// Typically this is *openai.FunctionDefinition for function calling.
	Definition() any
}

// ToolCall represents a single tool invocation emitted by the model.
type ToolCall interface {
	// Index returns the zero-based position of this tool call in a sequence of calls.
	Index() int

	// ID returns the unique identifier of this tool call.
	ID() string

	// Type returns the category of the tool call (e.g., "function").
	Type() string

	// Function returns details of the function call (name and arguments).
	Function() FunctionCall
}

// FunctionCall contains the details of a function-style tool invocation.
type FunctionCall interface {
	// Name returns the human-readable name of the tool being called.
	Name() string

	// Arguments returns the serialized arguments passed to the tool.
	Arguments() string
}

type tool struct {
	type_      string
	definition any
}

// Type implements Tool.
func (t *tool) Type() string {
	return t.type_
}

// Definition implements Tool.
func (t *tool) Definition() any {
	return t.definition
}

// toolcall is the internal implementation of ToolCall for function-style tools.
type toolcall struct {
	// index is the zero-based order of the call in the response.
	index int
	// id is the unique identifier assigned by the LLM to this call.
	id string
	// type_ is the textual type/category (e.g., "function").
	type_ string
	// fcall holds the concrete function call details (name and args).
	fcall funcall
}

// Index implements ToolCall.
func (tcall *toolcall) Index() int {
	return tcall.index
}

// ID implements ToolCall.
func (tcall *toolcall) ID() string {
	return tcall.id
}

// Type implements ToolCall.
func (tcall *toolcall) Type() string {
	return tcall.type_
}

// Function implements ToolCall.
func (tcall *toolcall) Function() FunctionCall {
	return &tcall.fcall
}

// funcall accumulates the function call arguments, supporting both
// complete argument payloads and incremental streaming deltas.
type funcall struct {
	// name is the function/tool name.
	name string
	// args holds the complete serialized arguments when provided at once.
	args string
	// buff accumulates streamed argument deltas until completion.
	buff strings.Builder
}

// Name implements FunctionCall.
func (fcall *funcall) Name() string {
	return fcall.name
}

// Arguments implements FunctionCall, returning the complete argument payload
// if present; otherwise returns the accumulated streamed content.
func (fcall *funcall) Arguments() string {
	if fcall.args != "" {
		return fcall.args
	}
	return fcall.buff.String()
}

// writeArgs appends an incremental delta to the argument buffer during streaming.
func (fcall *funcall) writeArgs(delta string) {
	fcall.buff.WriteString(delta)
}
