import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, set_seed




org="jmhuerta"
model_ckpt = "codeparrot"
dataset_name = 'transformersbook/codeparrot'

model = AutoModelForCausalLM.from_pretrained("./")
model.save_pretrained(model_ckpt, push_to_hub=True, organization=org)





'''
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer),gradient_checkpointing= True,)
model = AutoModelForCausalLM.from_config(config)


print(f'GPT-2 (xl) size: {model_size(model)/1000**2:.1f}M parameters')



model.save_pretrained(model_ckpt, push_to_hub=True, organization=org)



                    

org="jmhuerta"
model_ckpt = "codeparrot"
dataset_name = 'transformersbook/codeparrot'

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer),gradient_checkpointing= True,)
model = AutoModelForCausalLM.from_config(config)


print(f'GPT-2 (xl) size: {model_size(model)/1000**2:.1f}M parameters')



model.save_pretrained(model_ckpt, push_to_hub=True, organization=org)


# Commented parameters correspond to the small model
config = {"train_batch_size": 4, # 12, was 2
          "valid_batch_size": 2, # 12, this is the validation batchsize
          "weight_decay": 0.1,
          "shuffle_buffer": 1000,
          "learning_rate": 2e-4, # 5e-4
          "lr_scheduler_type": "cosine",
          "num_warmup_steps": 750, # 2000
          "gradient_accumulation_steps": 16, # 16
          "max_train_steps": 5000, # 50000, 150000
          "max_eval_steps": 2000,  # -1,
          "seq_length": 1024,
          "seed": 1,
          "save_checkpoint_steps": 10000} # 15000

args = Namespace(**config)


set_seed(args.seed)


project_name="jmhuerta/codeparrot


'''
