package openllm

import (
	"errors"
)

var (
	ErrEmptyChoices = errors.New("empty choices from completion response")
)
