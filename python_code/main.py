import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

model_name = 'ibm/labradorite-13b'
#model_name = '/lmsys/vicuna-7b-v1.5'

model = AutoModelForCausalLM.from_pretrained(model_name)

#tokenizer = AutoTokenizer.from_pretrained(model_name)

for name, param in model.named_parameters():
    print(name)
