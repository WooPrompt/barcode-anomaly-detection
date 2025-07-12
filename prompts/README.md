# Prompts Directory Structure

## Purpose
This directory contains all AI interaction configurations, protocols, and automation guides for the barcode anomaly detection project.

## Directory Structure

```
prompts/
├── project/                    # Static project context
│   └── principle.llm.txt      # Project goals, environment, data schema
├── protocol/                  # AI behavior rules
│   ├── learning_v1.llm.txt    # Learning-focused interaction style
│   └── analysis_log_behavior.llm.txt  # Systematic inquiry protocol
├── tasks/                     # Task-specific configurations
│   └── anomaly_detection/     # Anomaly detection modularization
│       ├── anomaly_detection.json     # Main task configuration
│       ├── edge.txt                   # Edge cases and requirements
│       ├── question_loop.llm.txt      # Systematic questioning protocol
│       └── Refactoring_Workflow.llm.txt  # Benchmarking workflow
├── engineering/               # Meta-level automation guidance
│   └── automation_guide.txt   # How to create new automation prompts
└── log/                       # Conversation analysis history
    └── [timestamped analysis files]
```

## Usage

### For AI Interactions:
1. **Load project context**: Always reference `project/principle.llm.txt`
2. **Follow protocols**: Apply rules from `protocol/` files
3. **Task execution**: Use configurations from `tasks/[task_name]/`
4. **Create analysis**: Save conversation logs in `log/`

### For New Automation:
1. Read `engineering/automation_guide.txt` for patterns
2. Create new task folder under `tasks/`
3. Follow established JSON command structure
4. Document decisions in analysis log files

### JSON Command Format:
```json
{
  "task": "task_name",
  "load_context_files": ["project/principle.llm.txt"],
  "protocol_context": ["protocol/learning_v1.llm.txt"],
  "analysis_reference": "log/task_analysis_YYYYMMDD_HHMM.txt",
  "requirements": {...}
}
```

## File Naming Conventions

- **Project files**: `principle.llm.txt`, `schema.llm.txt`
- **Protocol files**: `[behavior_name]_v[version].llm.txt`
- **Task configs**: `[task_name].json`
- **Analysis logs**: `[topic]_analysis_YYYYMMDD_HHMM.txt`
- **Documentation**: `README.md`, `[topic]_guide.txt`

## Benefits of This Organization

### Separation of Concerns:
- **project/**: What the project is (static context)
- **protocol/**: How AI should behave (interaction rules)
- **tasks/**: What to do (specific work configurations)
- **engineering/**: How to create new automations (meta-guidance)
- **log/**: What happened (historical decisions)

### Scalability:
- Easy to add new task types
- Protocol files are reusable across tasks
- Clear upgrade path for versions
- Knowledge accumulation in log files

### Maintainability:
- Clear file purposes
- Logical grouping
- Version tracking
- Decision history preservation

## Getting Started

1. **New AI session**: Load `project/principle.llm.txt` for context
2. **Follow protocols**: Apply `protocol/learning_v1.llm.txt` for interaction style
3. **Specific task**: Use relevant `tasks/[task_name]/` configuration
4. **Document decisions**: Create analysis file in `log/`

This structure enables systematic, reproducible AI interactions while building project knowledge over time.