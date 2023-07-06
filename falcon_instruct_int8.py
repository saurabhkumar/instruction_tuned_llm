# import the necessary packages

from transformers import (AutoModelForCausalLM, 
                          AutoConfig, 
                          AutoTokenizer, 
                          BitsAndBytesConfig,
                          GenerationConfig)
import torch
import datetime


model_path = "tiiuae/falcon-7b-instruct" # if model is loaded from huggingface hub
# Model download will take some time. It will be cached for later use.

# You can also use gitlfs to get the files - be aware of the large file sizes
# (or download them manually if thats the only possibility on your system)
#model_path = "full_path_to_model" # if model is stored locally, point to the folder

# If model is loaded from huggingface hub, provide a commit hash for safety
# Get the correct hash from huggingface hub
# https://huggingface.co/tiiuae/falcon-7b-instruct/tree/main

commit_hash = "c7f670a03d987254220f343c6b026ea0c5147185" 

tokenizer = AutoTokenizer.from_pretrained(model_path)

config = AutoConfig.from_pretrained(model_path, 
                                    trust_remote_code=True,
                                    revision=commit_hash, 
                                    pad_token_id=tokenizer.eos_token_id)

# Load the model in 8bit
model = AutoModelForCausalLM.from_pretrained(model_path,
        quantization_config=BitsAndBytesConfig(
        load_in_8bit=True),
        trust_remote_code=True,
        revision=commit_hash,
        torch_dtype=torch.float16,
        device_map={"": 0},
        pad_token_id=tokenizer.eos_token_id
        )

## The information about the course below is taken from the website 
## https://www.hs-augsburg.de/Informatik/Business-Information-Systems.html?mwg_rnd=7932405
## (The english translated version in browser (microsoft edge))

input_text = '''
Using only the information below about the course at the university, formulate 10 possible questions from prospective students.

Course: 
Business Information Systems (MSc)
Study content:
The course promotes dealing with complex issues and enables you to implement both economic and information technology concepts.

In terms of content, the course is divided into four strands of modules:
Scientific Fundamentals:
Principles of scientific and team-oriented work and application of mathematical models to solve problems.
Business Analysis and Modeling:
From the analysis and modeling of operational processes to requirements engineering to digital business models and their representation.
Business Application Systems:
From the conception and use of operative systems in the company to the processing and analysis of the data as a basis for decision-making.
IT Management:
From the implementation and management of individual projects in the IT environment to the short to long-term planning of the company-wide IT infrastructure.

Possible questions:
'''

input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
print('number of input tokens: {}'.format(len(input_ids[0])))

generation_config = GenerationConfig(
    temperature=0.3,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False,
    num_return_sequences=1,
    max_new_tokens=256
)

print('Starting generation\n')
time1 = datetime.datetime.now()
outputs = model.generate(input_ids=input_ids, 
                         generation_config=generation_config)

# The generated text has input text. We can skip this.
# In case of other models, there could be specific identifiers e.g. ### Response:
# This is not so in the case of falcon_instruct. We can simply skip thgis input text.

output_new_tokens = outputs[0][len(input_ids[0])-1:]

# For falcon model, the generate tends to create multiple <|endoftext|> tokens. 
# The reason for this is not clear and the model authors have to provide more details.

# A workaround is to use the generated text only till the first <|endoftext|> token
# <|endoftext|> token has token id 11
endoftext_token_id = 11

last_pos = output_new_tokens.tolist().index(endoftext_token_id)
output_new_tokens = output_new_tokens[:last_pos]

print(tokenizer.decode(output_new_tokens))
time2 = datetime.datetime.now()
print('\nTime taken for generation: {}'.format((time2-time1).seconds))

