# Instruction Tuned Language Models

This repository contains slides and code for a session on Instruction Tuned Language Models at the Technical University of Applied Science, Augsburg, Germany.
The session was done as an invited lecture within the NLP course taught by Prof. Alessandra Zarcone.

Note: The code samples and descriptions here are just to help the students understand instruction tuned language models. They are not intended for being used in any projects. There is no guarantee of correctness or usability. 

## The motivation
There has been a raging debate about the potentials and risks of AI, with even 'existential risks for humanity' being mentioned. Recently, even the most famous researchers in the domain have started taking sides in this debate. Understanding the basis of the opinions of these researchers is not easy for most students. 

It is undeniable that ChatGPT has been the trigger for this debate. We do not engage in the discussion here whether Large Language Models have the potential of understanding the world or whether they are just stochastic parrots. We leave that to the more knowledgable people :smirk: 

We try to address much simpler problems for a student at the University of Applied Sciences:
* How can I better understand how models like ChatGPT are developed?
* Is there a possibility that I could develop a smaller version of such a model and understand whether or not a small model can perform some tasks at par with a much larger model? 
* What is the nature of data that is used to train such models and what is the impact of the quality of data on the performance of such models?
* If *asking the question in the right form* is so important, can I experiment with sophisticated techniques like [*Chain of Thought*](https://arxiv.org/abs/2201.11903) or [*ReAct*](https://arxiv.org/abs/2210.03629) using my small model?

## The demonstration
We needed a model that can satisfy the following needs:
* The model weights can be downloaded without constraints
* An instruction tuned version of the model is available to validate if we can make progress with finetuning on another dataset
* The feasibility of finetuning the model on a GPU with 24GB VRAM (assuming some gaming GPUs can be used for betterment of science and humanity :grin: )

Amongst a few options available, we just decided to go with the [Falcon large language model](https://falconllm.tii.ae/). The model has been [shown to do well on multiple benchmarks](https://huggingface.co/blog/falcon). It can be easily used with [Hugging Face Transformers library](https://huggingface.co/docs/transformers/index) with [Parameter Efficient Finetuning (PEFT)](https://huggingface.co/blog/peft) and [Quantization (bitsandbytes)](https://huggingface.co/blog/hf-bitsandbytes-integration). However, students can experiment with other models.
The two model variants used:
* The base model: [falcon-7b](https://huggingface.co/tiiuae/falcon-7b)
* The instruction tuned model: [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)

## The model finetuning
We did not want to start from scratch with the finetuning code. The [*Alpaca* model](https://crfm.stanford.edu/2023/03/13/alpaca.html?mwg_rnd=9978114) has already triggered creation of multiple open source projects. However, these are dependant on using the [LLaMA model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/). We can use one such project - [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) and adapt it for being used with Falcon. We used a version of the [databricks-dolly-15k](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) dataset [adapted to suit the Alpaca format](https://huggingface.co/datasets/c-s-ale/dolly-15k-instruction-alpaca-format) as required by *Alpaca-Lora*.

### Steps for students to run their finetuning
These instructions are not meant to enable you to copy and paste commands to a terminal. Use the descriptions below to figure out the intermediate steps (you should be able to succesfully do this). This is a part of the learning process to be able to work with such models.

(I could not resist putting some comments here. So just for fun: When the LLM could itself decide to read these instructions, understand them, generate datasets and execute the code, it might be able to create offsprings to suit its goals :smiling_imp: ).

* Set up the environment (python dependencies) for working with falcon. Remember it uses some special features like flash attention requiring a specific version of PyTorch. Follow the instructions [here](https://huggingface.co/tiiuae/falcon-7b). You would need additional packages for PEFT and Quantization.

* Clone the *Alpaca-Lora* repository

After cloning the repository, you need the following adaptations to the code to make it work with falcon:
* In the *train* function in *finetune.py* specify *base_model* as *"tiiuae/falcon-7b"*, *data_path* as *"c-s-ale/dolly-15k-instruction-alpaca-format"* and the list for *lora_target_modules* as ["query","value"]. Specify *output_dir* to something like *"./falcon-loara"*
* Replace the use of *LlamaForCausalLM* with *AutoModelForCausalLM* and *LlamaTokenizer* with *AutoTokenizer*.
* There is a bug in the *Alpaca-Lora* code that prevents saving of the adapter weights. Read the comments for the [issue](https://github.com/tloen/alpaca-lora/issues/446) already logged and fix it (code commenting).
* In *utils/prompter.py*, modify *get_response* function to use "<|endoftext|>" token to find the end of the generated text and use the text only upto the first occurence of the token.

**Perform finetuning** as described in the instructions in the *Alpaca-Lora* repository (**remember to use the right values for the arguments** - base model, rank, output path etc..).

Important: If you are using a remote login using ssh at the university (for accessing a workstation or server), try to use *nohup* and an *&* at the end of the finetuning command to ensure that it keeps running when your ssh session is terminated. You can use the `tail -f nohup.out` in the same path where you started your finetuning script to see the progress (it might take days or hours based on the GPU and the number of epochs).

After finetuning is completed, **modify the *generate.py* as required for use with falcon** - set the correct base model and load the peft weights from the folder where you saved the finetuning weights, and use *AutoModelForCausalLM* and *AutoTokenizer* . (Hint: You might not necessarily need the setup with *gradio* as provided in alpaca-lora code. Just use the part of code needed for text generation - base model loading, PEFT parameter loading, GenerationConfig, and prompter modified for use with falcon. 

## Have fun and keep learning.
