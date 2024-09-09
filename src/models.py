import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# models/turboderp_Mixtral-8x7B-instruct-exl2_4.0bpw
# models/KnowMedPhi3-mini-it


def get_answer_checkpoint(prompt, model, tokenizer):    
    # if prompt does not end with \n\n, add it
    if not prompt.endswith('\n'):
        prompt += '\n'

    inputs = tokenizer([prompt], return_tensors="pt")

    generate_params = {
        'max_new_tokens': 200, 
        'temperature': 0.1, 
        'top_p': 1, 'top_k': 0, 
        'repetition_penalty': 1.15, 
        'typical_p': 1.0, 
        'guidance_scale': 1.0, 
        'penalty_alpha': 0,
        'encoder_repetition_penalty': 1, 
        'no_repeat_ngram_size': 0, 
        'do_sample': True, 
        'use_cache': True,
        'eos_token_id': [32007] # tokenizer.eos_token_id
    }
    
    generate_params['inputs'] = inputs['input_ids'].cuda()

    output = model.generate(**generate_params)[0]

    output_gen = output[inputs['input_ids'].shape[1]:]
    output_text = tokenizer.decode(output_gen)
    #output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_text


def load_model_tokenizer(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = 32000
    tokenizer.eos_token_id = 32007
    #tokenizer.pad_token_id = tokenizer.eos_token_id

    params_model = {
        'low_cpu_mem_usage': True, 
        'torch_dtype': torch.float16, 
        'trust_remote_code': True
    }

    from src.exllamav2_hf import Exllamav2HF
    if 'exl2' in checkpoint_path:
        model_name = checkpoint_path.split('/')[-1]
        model = Exllamav2HF.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            **params_model
        )

    model = model.cuda()
    print('model loaded!')
    return model, tokenizer