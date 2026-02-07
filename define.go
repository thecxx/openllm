package openllm

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"

	"github.com/sashabaranov/go-openai/jsonschema"
	"github.com/thecxx/openllm/constants"
)

var (
	TypeContext = reflect.TypeOf((*context.Context)(nil)).Elem()
)

// FunctionOptions holds the configuration options for a function tool.
type FunctionOptions struct {
	name        string
	description string
	fn          any
	parameters  any
	strict      bool
}

// function is the intermediate structure for a tool's definition.
type function struct {
	name        string
	description string
	fn          any
	parameters  any
	strict      bool
}

// FunctionOption defines a functional option for configuring a function tool.
type FunctionOption func(opts *FunctionOptions)

// WithFunction sets a callback function for the tool.
// The parameter struct T should use `openllm` for parameter configuration.
// Format: `openllm:"name,required,desc=..."`
func WithFunction(fn any) FunctionOption {
	return func(opts *FunctionOptions) { opts.fn = fn }
}

// WithFunctionParameters sets the schema that describes the function's parameters.
func WithFunctionParameters(parameters any) FunctionOption {
	return func(opts *FunctionOptions) { opts.parameters = parameters }
}

// WithFunctionStrict enables or disables Strict Mode for structured output.
func WithFunctionStrict(strict bool) FunctionOption {
	return func(opts *FunctionOptions) { opts.strict = strict }
}

// DefineFunction creates a generic function tool definition.
func DefineFunction(name, description string, opts ...FunctionOption) Tool {
	options := &FunctionOptions{
		name:        name,
		description: description,
	}

	for _, opt := range opts {
		opt(options)
	}

	if options.parameters == nil && options.fn != nil {
		parameters := generateParametersFromFunc(options.fn)
		if parameters != nil {
			options.parameters = *parameters
		}
	}

	// Ensure Parameters is not nil to prevent API validation errors.
	if options.parameters == nil {
		options.parameters = jsonschema.Definition{
			Type:       jsonschema.Object,
			Properties: make(map[string]jsonschema.Definition),
			Required:   make([]string, 0),
		}
	} else {
		// Normalize parameters to jsonschema.Definition if possible
		if _, ok := options.parameters.(jsonschema.Definition); !ok {
			data, err := json.Marshal(options.parameters)
			if err == nil {
				var def jsonschema.Definition
				if err := json.Unmarshal(data, &def); err == nil && def.Type != "" {
					options.parameters = def
				} else {
					options.parameters = jsonschema.Definition{
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
		definition: &function{
			name:        options.name,
			description: options.description,
			fn:          options.fn,
			parameters:  options.parameters,
			strict:      options.strict,
		},
	}
}

// generateParametersFromFunc analyzes the function signature and generates a JSON Schema.
// It supports functions with (context.Context, struct/ptr) or just (struct/ptr).
func generateParametersFromFunc(fn any) *jsonschema.Definition {
	if fn == nil {
		return nil
	}

	t := reflect.TypeOf(fn)
	if t.Kind() != reflect.Func {
		return nil
	}

	numIn := t.NumIn()
	if numIn == 0 {
		return nil
	}

	// Determine which argument is the parameter struct
	paramType := t.In(0)

	if paramType.Implements(TypeContext) {
		if numIn < 2 {
			return nil
		}
		paramType = t.In(1)
	}

	// Unwrap pointer if necessary
	if paramType.Kind() == reflect.Ptr {
		paramType = paramType.Elem()
	}

	if paramType.Kind() != reflect.Struct {
		return nil
	}

	return parseStructToDefinition(paramType)
}

// parseStructToDefinition recursively converts a Go struct type into a JSON Schema definition.
// It inspects struct fields for the 'openllm' tag to determine property names,
// descriptions, and validation requirements.
//
// Key Features:
//   - Type Mapping: Maps Go primitives to JSON Schema types (string, integer, number, boolean).
//   - Recursion: Handles nested structs by generating nested JSON objects.
//   - Collections: Automatically detects slices and arrays, generating 'array' schemas with 'items'.
//   - Pointer Resolution: Transparently unwraps pointers to both structs and primitive types.
//   - Visibility: Respects Go's visibility rules, skipping unexported (private) fields.
//
// Tag Syntax: `openllm:"name,required,desc=Example description"`
func parseStructToDefinition(t reflect.Type) *jsonschema.Definition {
	// Base case for recursion: handle pointer types immediately
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	def := &jsonschema.Definition{
		Type:       jsonschema.Object,
		Properties: make(map[string]jsonschema.Definition),
		Required:   []string{},
	}

	if t.Kind() != reflect.Struct {
		return def
	}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if field.PkgPath != "" { // Skip unexported fields
			continue
		}

		tag := field.Tag.Get("openllm")
		if tag == "" || tag == "-" {
			continue
		}

		name, opts := parseTag(tag)
		fieldDef := getJSONType(field.Type)
		fieldDef.Description = opts["desc"]

		// Support nested structs
		if field.Type.Kind() == reflect.Struct ||
			(field.Type.Kind() == reflect.Ptr && field.Type.Elem().Kind() == reflect.Struct) {
			nested := parseStructToDefinition(field.Type)
			nested.Description = fieldDef.Description // Carry over description
			fieldDef = *nested
		}

		def.Properties[name] = fieldDef
		if _, ok := opts["required"]; ok {
			def.Required = append(def.Required, name)
		}
	}
	return def
}

// Helper: Maps Go Kind to JSON Schema Type
func getJSONType(t reflect.Type) jsonschema.Definition {
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.String:
		return jsonschema.Definition{Type: jsonschema.String}
	case reflect.Bool:
		return jsonschema.Definition{Type: jsonschema.Boolean}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return jsonschema.Definition{Type: jsonschema.Integer}
	case reflect.Float32, reflect.Float64:
		return jsonschema.Definition{Type: jsonschema.Number}
	case reflect.Slice, reflect.Array:
		items := getJSONType(t.Elem())
		return jsonschema.Definition{Type: jsonschema.Array, Items: &items}
	default:
		return jsonschema.Definition{Type: jsonschema.Object}
	}
}

// Helper: Flexible tag parsing
func parseTag(tag string) (string, map[string]string) {
	parts := strings.Split(tag, ",")
	res := make(map[string]string)
	for _, p := range parts[1:] {
		if k, v, found := strings.Cut(p, "="); found {
			res[k] = v
		} else {
			res[p] = ""
		}
	}
	return parts[0], res
}
