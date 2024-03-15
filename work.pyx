import cython
import copy
from cython.view cimport array

cpdef generate_sequences_cython(list[] chain, int last_hidden_state, float[:,:] probs,int start_idx, int[:] tokens,int token_pos):
    cdef int idx, token_curr
    cdef float prob
    cdef list values_for_new_chain
    if start_idx >= last_hidden_state or token_pos >= len(tokens):
        if len(chain) > 2:
            return chain

    for idx in range(start_idx, last_hidden_state):
        token_curr = tokens[token_pos]
        prob = probs[idx][token_curr]
        if prob >= 0.05:
            values_for_new_chain = [[prob], [token_pos]]

            if len(chain) == 1:
                chain.append(values_for_new_chain)
            else:
                current_chain = copy.deepcopy(chain[-1])
                for i in range(len(current_chain)):
                    current_chain[i].append(values_for_new_chain[i][0])

                chain.append(current_chain)
            generate_sequences_cython(chain, last_hidden_state, probs,
                                      idx+1, tokens, token_pos+1)
        else:
            if len(chain) > 2:
                return chain