import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, set_seed
import re
from transformers import set_seed 
from transformers import pipeline, set_seed



org="jmhuerta"
model_ckpt = "codeparrot"
dataset_name = 'transformersbook/codeparrot'

#model = AutoModelForCausalLM.from_pretrained("./")
#model.save_pretrained(model_ckpt, push_to_hub=True, organization=org)

#model_ckpt = 'transformersbook/codeparrot-small'
model_ckpt = 'codeparrot'

generation = pipeline('text-generation', model=model_ckpt, device=0)



def first_block(string):
    return re.split('\nclass|\ndef|\n#|\n@|\nprint|\nif', string)[0].rstrip()

def complete_code(pipe, prompt, max_length=64, num_completions=4, seed=1):
    set_seed(seed)
    print("Prompt is:",prompt)

    gen_kwargs = {"temperature":0.4, "top_p":0.95, "top_k":0, "num_beams":1,
                  "do_sample":True,}
    code_gens = generation(prompt, num_return_sequences=num_completions, 
                            max_length=max_length, **gen_kwargs)
    code_strings = []
    for code_gen in code_gens:
        generated_code = first_block(code_gen['generated_text'][len(prompt):])
        code_strings.append(generated_code)
    print(('\n'+'='*80 + '\n').join(code_strings))



prompt = '''def hello_world(a: int):
    """Print the string Hello World a times."""'''

complete_code(generation, prompt,max_length=24)



prompt = '''def area_of_rectangle(a: float, b: float):
    """Return the area of the rectangle."""'''

complete_code(generation, prompt,max_length=64)


prompt = '''X = np.random.randn(100, 100)
y = np.random.randint(0, 1, 100)

# fit random forest classifier with 20 estimators'''

complete_code(generation, prompt, max_length=96)




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
