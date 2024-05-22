import numpy as np


def find_indices(prompt, objects, model):
    support_multiword = False
    
    # Search for the word in the converted tokens_ids
    if not support_multiword:
        prompt_split = prompt.split()
        prompt_tokens = [model.tokenizer.convert_ids_to_tokens(x) for x in model.tokenizer.encode(prompt_split)]
        
        tokens_indices = []
        for token in objects:
            if token not in prompt_tokens:
                raise ValueError("couldn't find token in prompt, probably split into multiple tokens")
            token_index = [i for i, prompt_token in enumerate(prompt_tokens) if prompt_token == token]
            tokens_indices.append(token_index)
            
    # Search for sub-list in larger list of ids
    else:
        tokens_indices = []
        prompt_encoded = np.array(model.tokenizer.encode(prompt))
        for object in objects:
            
            object_tokens_ids = np.array(model.tokenizer.encode(object)[1:-1])
            start_idx = find_sub_np_array_in_larger_array(large_array=prompt_encoded, small_array=object_tokens_ids)
            indices = [start_idx + i for i in range(object_tokens_ids.shape[0])]
            tokens_indices.append(indices)
            
    return tokens_indices


def find_sub_np_array_in_larger_array(large_array, small_array):
    """
    Based on https://stackoverflow.com/questions/46806973/numpy-search-for-an-array-a-with-same-pattern-in-a-larger-array-b
    """
    
    return large_array.tostring().index(small_array.tostring())//large_array.itemsize   
