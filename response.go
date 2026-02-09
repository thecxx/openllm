package openllm

import "time"

// Response wraps the final assistant message and any tool calls produced by the model.
// Both blocking and streaming APIs return a Response upon completion.
type Response interface {
	// Answer returns the final assistant message after generation finishes.
	Answer() Message
	// ToolCalls returns tool invocation records in the order they were produced.
	ToolCalls() []ToolCall
	// Meta returns the request metadata (provider, model, request ID, etc.).
	Meta() Meta
	// Duration returns the total elapsed time of the request.
	Duration() time.Duration
}

// response is the concrete implementation of Response.
type response struct {
	// answer is the final assistant message constructed from the model output.
	answer Message
	// tcalls holds all function tool calls captured during generation.
	tcalls []ToolCall
	// meta contains request metadata.
	meta Meta
	// duration captures the elapsed time from request start to completion.
	duration time.Duration
}

// Answer implements Response by returning the final assistant message.
func (resp *response) Answer() Message {
	return resp.answer
}

// ToolCalls implements Response by returning the collected tool calls.
func (resp *response) ToolCalls() []ToolCall {
	return resp.tcalls
}

// Meta implements Response.
func (resp *response) Meta() Meta {
	return resp.meta
}

// Duration implements Response.
func (resp *response) Duration() time.Duration {
	return resp.duration
}

// Meta contains request metadata:
type Meta struct {
	// backend provider (e.g., openai, anthropic).
	Provider string
	// model name.
	Model string
	// request ID (useful for troubleshooting/auditing).
	RequestID string
	// (OpenAI) server fingerprint to distinguish backend versions.
	SystemFingerprint string
	// reason the generation stopped (e.g., stop_sequence, max_tokens, tool_use).
	StopReason string
}
