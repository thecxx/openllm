package openllm

// ChatOption represents a functional option to configure a single chat request.
// Options are applied in order and only affect the specific call where they are passed.
type ChatOption func(*ChatOptions)

// ChatOptions holds per-request configuration used to build the OpenAI chat completion.
// Fields are intentionally unexported; use With* helpers to set them.
type ChatOptions struct {
	// prompt is the system prompt included at the beginning of the conversation.
	prompt string
	// tools is the list of function tools available for the model to call.
	tools []Tool
	// watcher handles streaming events during ChatCompletionStream; ignored for blocking calls.
	watcher StreamWatcher
	// temperature controls randomness; nil leaves it to server defaults.
	// Typical values range from 0.0 (deterministic) to 2.0 (more random).
	temperature *float32
}

// WithSystemPrompt sets the system prompt for the current chat request.
func WithSystemPrompt(prompt string) ChatOption {
	return func(opts *ChatOptions) { opts.prompt = prompt }
}

// WithTools sets the function tools the model may call during generation.
func WithTools(tools []Tool) ChatOption {
	return func(opts *ChatOptions) { opts.tools = tools }
}

// WithTool set a function tool the model may call during generation.
func WithTool(tool Tool) ChatOption {
	return func(opts *ChatOptions) { opts.tools = append(opts.tools, tool) }
}

// StreamWatcher sets the handler used to receive streamed deltas and tool-call updates.
func WithStreamWatcher(watcher StreamWatcher) ChatOption {
	return func(opts *ChatOptions) { opts.watcher = watcher }
}

// WithTemperature sets temperature for the current request; if not provided, server defaults apply.
func WithTemperature(temperature float32) ChatOption {
	return func(opts *ChatOptions) { opts.temperature = &temperature }
}
