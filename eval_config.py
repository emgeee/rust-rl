models:
  local:
    - name: "test-local"
      model_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
      device: "mps"
  
  anthropic:
    - name: "claude-3-5-sonnet"
      model_id: "claude-3-5-sonnet-20241022"
    - name: "claude-3-5-haiku"
      model_id: "claude-3-5-haiku-20241022"
    - name: "claude-3-opus"
      model_id: "claude-3-opus-20240229"
  
  openai:
    - name: "gpt-4o"
      model_id: "gpt-4o"
    - name: "gpt-4o-mini"
      model_id: "gpt-4o-mini"
    - name: "gpt-4-turbo"
      model_id: "gpt-4-turbo"
    - name: "o1-preview"
      model_id: "o1-preview"
    - name: "o1-mini"
      model_id: "o1-mini"
  
  grok:
    - name: "grok-2"
      model_id: "grok-2-1212"
    - name: "grok-2-mini"
      model_id: "grok-2-mini"

generation_params:
  max_new_tokens: 256
  temperature: 0.2
  top_p: 0.9

dataset:
  path: "qwen3-rust-finetune/test_eval.parquet"

output:
  base_dir: "qwen3-rust-finetune/outputs/test"

evaluation:
  tools: ["build", "clippy", "test"]
  save_every: 2
