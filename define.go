package openllm

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"

	"github.com/sashabaranov/go-openai/jsonschema"
	"github.com/thecxx/openllm/constants"
)

// FunctionOptions holds the configuration options for a function tool.
type FunctionOptions struct {
	Name        string
	Description string
	InvokeFunc  any
	Parameters  any
	Strict      bool
}

// FunctionDefinition is the intermediate structure for a tool's definition.
type FunctionDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
	Strict      bool   `json:"strict,omitempty"`
	InvokeFunc  any    `json:"-"`
}

// FunctionOption defines a functional option for configuring a function tool.
type FunctionOption func(opts *FunctionOptions)

// WithFunction sets a callback function for the tool.
func WithFunction(fnptr any) FunctionOption {
	return func(opts *FunctionOptions) { opts.InvokeFunc = fnptr }
}

// WithFunctionParameters sets the schema that describes the function's parameters.
func WithFunctionParameters(parameters any) FunctionOption {
	return func(opts *FunctionOptions) { opts.Parameters = parameters }
}

// WithFunctionStrict enables or disables Strict Mode for structured output.
func WithFunctionStrict(strict bool) FunctionOption {
	return func(opts *FunctionOptions) { opts.Strict = strict }
}

// DefineFunction creates a generic function tool definition.
func DefineFunction(name, description string, opts ...FunctionOption) Tool {
	options := &FunctionOptions{
		Name:        name,
		Description: description,
	}

	for _, opt := range opts {
		opt(options)
	}

	if options.Parameters == nil && options.InvokeFunc != nil {
		parameters := generateParametersFromFunc(options.InvokeFunc)
		if parameters != nil {
			options.Parameters = *parameters
		}
	}

	// Ensure Parameters is not nil to prevent API validation errors.
	if options.Parameters == nil {
		options.Parameters = jsonschema.Definition{
			Type:       jsonschema.Object,
			Properties: make(map[string]jsonschema.Definition),
			Required:   make([]string, 0),
		}
	} else {
		// Normalize parameters to jsonschema.Definition if possible
		if _, ok := options.Parameters.(jsonschema.Definition); !ok {
			data, err := json.Marshal(options.Parameters)
			if err == nil {
				var def jsonschema.Definition
				if err := json.Unmarshal(data, &def); err == nil && def.Type != "" {
					options.Parameters = def
				} else {
					options.Parameters = jsonschema.Definition{
						Type:       jsonschema.Object,
						Properties: make(map[string]jsonschema.Definition),
						Required:   make([]string, 0),
					}
				}
			}
		}
	}

	return &tool{
		type_: constants.ToolTypeFunction,
		definition: &FunctionDefinition{
			Name:        options.Name,
			Description: options.Description,
			Parameters:  options.Parameters,
			Strict:      options.Strict,
			InvokeFunc:  options.InvokeFunc,
		},
	}
}

// generateParametersFromFunc analyzes the signature of the provided function
// and generates a JSON Schema definition based on the parameter struct's tags.
func generateParametersFromFunc(fn any) *jsonschema.Definition {
	if fn == nil {
		return nil
	}

	typ := reflect.TypeOf(fn)
	if typ.Kind() != reflect.Func {
		return nil
	}

	// We expect the last or only argument to be the parameters struct (usually a pointer)
	var paramType reflect.Type
	numIn := typ.NumIn()
	if numIn == 0 {
		return nil
	}

	// Check if first arg is context.Context
	firstArg := typ.In(0)
	ctxInterface := reflect.TypeOf((*context.Context)(nil)).Elem()

	if firstArg.Implements(ctxInterface) {
		if numIn < 2 {
			return nil
		}
		paramType = typ.In(1)
	} else {
		paramType = typ.In(0)
	}

	// Ensure it's a struct or pointer to struct
	if paramType.Kind() == reflect.Ptr {
		paramType = paramType.Elem()
	}
	if paramType.Kind() != reflect.Struct {
		return nil
	}

	return parseStructToDefinition(paramType)
}

func parseStructToDefinition(t reflect.Type) *jsonschema.Definition {
	def := &jsonschema.Definition{
		Type:       jsonschema.Object,
		Properties: make(map[string]jsonschema.Definition),
		Required:   []string{},
	}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)

		// Skip unexported fields
		if field.PkgPath != "" {
			continue
		}

		argTag := field.Tag.Get("openllm")
		if argTag == "" {
			continue
		}

		parts := strings.Split(argTag, ",")

		var (
			name     = parts[0]
			required bool
			desc     string
		)
		for i := 1; i < len(parts); i++ {
			part := parts[i]
			if part == "required" {
				required = true
			} else if strings.HasPrefix(part, "desc=") {
				desc = strings.TrimPrefix(part, "desc=")
				break
			}
		}

		fieldDef := jsonschema.Definition{
			Description: desc,
		}

		// Map Go types to JSON Schema types
		switch field.Type.Kind() {
		case reflect.String:
			fieldDef.Type = jsonschema.String
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			fieldDef.Type = jsonschema.Integer
		case reflect.Float32, reflect.Float64:
			fieldDef.Type = jsonschema.Number
		case reflect.Bool:
			fieldDef.Type = jsonschema.Boolean
		case reflect.Struct:
			subDef := parseStructToDefinition(field.Type)
			fieldDef = *subDef
		case reflect.Ptr:
			if field.Type.Elem().Kind() == reflect.Struct {
				subDef := parseStructToDefinition(field.Type.Elem())
				fieldDef = *subDef
			}
		}

		def.Properties[name] = fieldDef
		if required {
			def.Required = append(def.Required, name)
		}
	}

	return def
}
