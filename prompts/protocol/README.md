# Protocol Documentation

## Overview

This folder contains AI behavior protocols that define **HOW** AI should interact during different types of tasks. Each protocol provides systematic approaches for consistent and effective AI assistance.

## Available Protocols

### Core Protocols

#### `learning_v1.llm.txt` - Educational Interaction
- **Purpose**: Step-by-step learning-focused development
- **When to use**: When user wants to understand code decisions and learn
- **Key features**:
  - Always ask where to insert new code and explain why
  - Provide 2-3 code options with pros/cons
  - Guide step-by-step in small, clear plans
  - Show full, readable examples
  - Help rollback safely and commit with confidence

#### `question_loop.llm.txt` - Systematic Questioning
- **Purpose**: Thorough task analysis before coding
- **When to use**: For any new complex task requiring analysis
- **Key features**:
  - 10-step questioning checklist
  - Identifies ambiguities and edge cases
  - Suggests multiple solution strategies
  - Creates question.txt files with knowns/unknowns
  - Prevents assumptions through structured inquiry

#### `analysis_log_behavior.llm.txt` - Decision Tracking
- **Purpose**: Context accumulation and persistent knowledge building
- **When to use**: For building cumulative project expertise
- **Key features**:
  - Systematic analysis and decision tracking
  - Methodical analyzer personality
  - Persistent knowledge building across sessions

#### `dynamic_question_protocol.llm.txt` - Automated Question Generation
- **Purpose**: Generate questions and edge cases from task definitions
- **When to use**: For incremental question/edge case generation
- **Key features**:
  - Reads task JSON files
  - Generates incremental questions and edge cases
  - Supports "Skip" mechanism
  - Appends to existing files

## How to Use Protocols

### Manual Application
Load any protocol file to apply its behavior:
```
@prompts/protocol/learning_v1.llm.txt
```

### Automated Application
Protocols are automatically loaded via the automation system:
```
@prompts/automation/init.json
```

### Protocol Combinations
Protocols can be combined for comprehensive analysis:
1. **question_loop.llm.txt** - Analyze the task
2. **learning_v1.llm.txt** - Educational implementation
3. **analysis_log_behavior.llm.txt** - Document decisions

## Integration with Automation

Protocols are integrated into the automation workflow:
- **init.json**: Loads protocols automatically at session start
- **Session lifecycle**: Applies protocols consistently
- **Metadata tracking**: Records protocol usage and effectiveness

## Best Practices

### For New Tasks
1. Start with `question_loop.llm.txt` for analysis
2. Apply `learning_v1.llm.txt` for educational coding
3. Use `analysis_log_behavior.llm.txt` for decision tracking

### For Routine Tasks
- Use automation system with pre-configured protocols
- Let `init.json` handle protocol loading

### For Research Tasks
- Use `dynamic_question_protocol.llm.txt` for question generation
- Combine with task-specific analysis files

## Protocol Evolution

Protocols follow the Meta→Templates→Tasks pattern:
- **Meta**: Design principles for creating protocols
- **Templates**: Reusable protocol templates
- **Tasks**: Filled protocol instances for specific domains

## Adding New Protocols

1. Design in `prompts/meta/`
2. Create template in `prompts/templates/`
3. Implement in `prompts/protocol/`
4. Update this README with usage instructions