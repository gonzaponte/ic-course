import numpy as np

def replace_tag(filename, tokens_in, token_out):
    if isinstance(tokens_in, str):
        # "a b" will turn into ["a", "b"]
        # "a" will turn into ["a"]
        tokens_in = tokens_in.split()

    for token_in in tokens_in:
        filename = filename.replace(token_in, token_out)
    return filename

def find_highest_wf(wfs, event_number):
    assert len(wfs.shape) == 3
    wfs   = wfs[event_number]
    index = np.argmax(np.max(wfs, axis=1))
    return wfs[index]