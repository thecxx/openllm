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
	// maxTokens limits the maximum number of tokens generated in the response.
	maxTokens *int
	// temperature controls randomness; nil leaves it to server defaults.
	// Typical values range from 0.0 (deterministic) to 2.0 (more random).
	temperature *float64
	// topK controls the number of highest probability vocabulary tokens to keep for top-k-filtering.
	topK *int
	// topP controls nucleus sampling, keeping the top tokens with cumulative probability >= topP.
	topP *float64
	// reasoningLevel controls the reasoning effort/budget.
	// Values should be one of "low", "medium", "high" (see constants/reasoning.go).
	reasoningLevel *string
}

// WithReasoning sets the reasoning level.
// For OpenAI o1/o3, this maps directly to `reasoning_effort`.
// For Anthropic Claude, this maps to a token budget (Low: 1024, Medium: 4096, High: 8192, capped by max_tokens).
func WithReasoning(level string) ChatOption {
	return func(opts *ChatOptions) { opts.reasoningLevel = &level }
}

// WithSystemPrompt sets the system prompt for the current chat request.
func WithSystemPrompt(prompt string) ChatOption {
	return func(opts *ChatOptions) { opts.prompt = prompt }
}

// WithTool sets the function tools the model may call during generation.
func WithTool(tools ...Tool) ChatOption {
	return func(opts *ChatOptions) { opts.tools = append(opts.tools, tools...) }
}

// StreamWatcher sets the handler used to receive streamed deltas and tool-call updates.
func WithStreamWatcher(watcher StreamWatcher) ChatOption {
	return func(opts *ChatOptions) { opts.watcher = watcher }
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(maxTokens int) ChatOption {
	return func(opts *ChatOptions) { opts.maxTokens = &maxTokens }
}

// WithTemperature sets temperature for the current request; if not provided, server defaults apply.
func WithTemperature(temperature float64) ChatOption {
	return func(opts *ChatOptions) { opts.temperature = &temperature }
}

// WithTopK sets the Top-K sampling parameter.
// Only the top K tokens with the highest probabilities are considered for generation.
func WithTopK(topK int) ChatOption {
	return func(opts *ChatOptions) { opts.topK = &topK }
}

// WithTopP sets the Top-P (nucleus) sampling parameter.
// Only the top tokens with cumulative probability >= topP are considered for generation.
func WithTopP(topP float64) ChatOption {
	return func(opts *ChatOptions) { opts.topP = &topP }
}
