# Automation System

Session lifecycle management for barcode anomaly detection project.

## Files

### Entry Point
- **`command.json`** - Session setup automation
  - Loads context files and protocols
  - Executes immediate_tasks, default_tasks, then main tasks
  - Handles metadata updates and lineage tracking

### Exit Point  
- **`ai_handoff.json`** - Task handoff maintenance
  - Updates ai_handoff.txt with current status
  - Logs decisions and maintains project context
  - Keeps documentation under 500 tokens

### Maintenance
- **`update_index.json`** - Directory structure updates
  - Scans project directories recursively
  - Updates index.llm.txt and README.md with current structure
  - Maintains AI-friendly documentation

### Planned
- **`directory_scan.json`** - Pure directory scanning utility

## Usage

### Session Lifecycle
```bash
# Start session
@automation/command.json

# Work on tasks...

# Exit session  
@automation/ai_handoff.json
```

### Manual Commands
```bash
# Update documentation
@automation/update_index.json

# Directory scanning
@automation/directory_scan.json
```

## Integration

The automation system works with the prompt framework:
- **Context**: Loads from `prompts/context/`
- **Protocols**: Applies from `prompts/protocol/`
- **Tasks**: Executes from `prompts/task/`
- **Logging**: Records to `prompts/log/`