package constants

import (
	openai "github.com/sashabaranov/go-openai"
)

const (
	RoleUser      = string(openai.ChatMessageRoleUser)
	RoleAssistant = string(openai.ChatMessageRoleAssistant)
	RoleSystem    = string(openai.ChatMessageRoleSystem)
	RoleTool      = string(openai.ChatMessageRoleTool)
)
