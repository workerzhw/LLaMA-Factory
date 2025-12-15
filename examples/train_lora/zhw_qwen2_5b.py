# 1. 安装轻量化依赖
# pip install peft bitsandbytes-cpu  # ARM版bitsandbytes（CPU量化）

# 2. 微调脚本（Python）
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model = AutoModelForCausalLM.from_pretrained(
  "Qwen/Qwen2-0.5B-Instruct",
  load_in_4bit=True,  # 4bit量化
  device_map="auto",
  torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

lora_config = LoraConfig(
  r=2, alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
  per_device_train_batch_size=1,
  gradient_accumulation_steps=8,
  learning_rate=1e-4,
  num_train_epochs=3,
  output_dir="./qwen_rpi_finetune",
  fp16=False,  # 树莓派无FP16，用FP32
  gradient_checkpointing=True  # 内存优化
)

trainer = SFTTrainer(
  model=model,
  args=training_args,
  train_dataset=your_dataset,  # 自定义小数据集
  tokenizer=tokenizer,
  max_seq_length=512
)
trainer.train()
