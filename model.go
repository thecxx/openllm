package openllm

import (
	"context"
)

// StreamWatcher handles events emitted during LLM generation.
type StreamWatcher interface {
	// OnRefusal is invoked when the model explicitly refuses to answer (e.g., safety filters).
	// The chunk parameter contains the partial refusal message.
	OnRefusal(chunk string) error

	// OnReasoning is invoked when the model emits "Chain of Thought" tokens.
	// The chunk parameter contains partial reasoning content (e.g., reasoning_content or thinking block).
	OnReasoning(chunk string) error

	// OnContent is invoked whenever the model emits a piece of the final visible response.
	// The chunk parameter contains the partial response text.
	OnContent(chunk string) error

	// OnToolCall is invoked when a tool call is detected.
	// The tcall contains tool metadata, and chunk contains the partial JSON arguments string.
	OnToolCall(ctx context.Context, tcall ToolCall, chunk string) (err error)

	// OnStop is invoked after the model has finished producing all output.
	OnStop() error
}

// Model defines the abstract interface for an LLM engine.
type Model interface {
	// Name returns the unique, human-readable name of the LLM core.
	Name() string

	// Description returns a brief description of the LLM core.
	Description() string

	// ChatCompletion performs a blocking chat completion request.
	// It takes a context for cancellation, a slice of messages as conversation history,
	// and optional ChatOption for configuration (e.g., tools, reasoning effort).
	// It returns the final aggregated Response and any execution error.
	ChatCompletion(ctx context.Context, messages []Message, opts ...ChatOption) (resp Response, err error)

	// ChatCompletionStream performs a streaming chat completion request.
	// It takes a context, conversation history, and ChatOption (which must include a StreamWatcher).
	// Partial outputs are pushed to the watcher; the returned Response contains final metadata.
	ChatCompletionStream(ctx context.Context, messages []Message, opts ...ChatOption) (resp Response, err error)
}
