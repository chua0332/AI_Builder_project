import math
import re

def long_chunk(text, context_length=512):
    '''
    Splits long chunks into chunks of at most 'context length' words
    '''
    words = re.findall(r'\w+', text)  # Tokenize words once
    total_words = len(words)
    repeats = math.ceil(total_words / context_length)  # Calculate how many chunks we need
    
    chunks = []
    tracker = 0  # Start index
    
    for i in range(repeats):
        b = (i + 1) * context_length  # Compute end index for chunk
        chunked_text = ' '.join(words[tracker:b])  # Join words into a chunk
        chunks.append(chunked_text)  # Store the chunk
        tracker = b  # Move tracker to next chunk
    
    return chunks  # Return all chunks as a list
