import numpy as np
import pandas as pd


# mininum edit distance algorithm or wagner fischer algorithm with traceback
def wagner_fischer(source, target):
    """
    Input:
        source: source word (the wrong one)
        target: target word (the correct one)
    Output:
        med: the minimum edit distance
        cost_matrix: a matrix contains the minium edit distance
        trace_matrix: a matrix records the trace
    The default costs are: insert cost = 1, delete cost = 1, replace cost = 2
    It is import to have the following matrix in your mind
      \#  H E L L O
    \# 0  1 2 3 4 5 \n
    S  1 \n 
    P  2 \n 
    E  3 \n 
    L  4 \n 
    L  5 \n 
    """
    m = len(source)
    n = len(target)
    
    # initialize the cost matrix
    cost_matrix = np.zeros((m+1, n+1), dtype=int)  
    
    # fill in the initial delete or insert cost from 0 to the length(word)
    cost_matrix[:, 0] = range(m+1)
    cost_matrix[0, :] = range(n+1)
    
    # initialize the traceback matrix
    trace_matrix = np.zeros((m+1, n+1),
                 dtype=[('ins', 'b'), ('del', 'b'), ('rep', 'b')])
    
    # fill in the initial trace: delete source and insert target
    trace_matrix[1:, 0] = (0, 1, 0)
    trace_matrix[0, 1:] = (1, 0, 0)
    
    for i, s in enumerate(source, start=1):
        for j, t in enumerate(target, start=1):
            del_cost = cost_matrix[i-1, j] + 1
            ins_cost = cost_matrix[i, j-1] + 1
            rep_cost = cost_matrix[i-1, j-1] + (0 if s == t else 2)
            
            mc = min(del_cost, ins_cost, rep_cost)
            
            # update trace matrix, the order has to be the same with the
            # dtype=[('ins', 'b'), ('del', 'b'), ('rep', 'b')]
            trace_matrix[i, j] = (ins_cost==mc, del_cost==mc, rep_cost==mc)
            
            # store the minimum edit distance
            cost_matrix[i, j] = mc
    
    med = cost_matrix[m, n]  # the last element as the initial index = 0 
    
    return med, cost_matrix, trace_matrix
    
    
def naive_trace(trace_matrix):
    """
    Input: a trace matrix that indicates the editing path with 
           dtype=[('ins', 'b'), ('del', 'b'), ('rep', 'b')]
    Output: a list of index of trace matrix 
    """
    i, j = trace_matrix.shape[0]-1, trace_matrix.shape[1]-1
    backtrace_indexs = [(i, j)]  # starting from the last cell
    
    # iterate until it reaches the starting point (0, 0)
    # BE CAREFULE: the order matters
    while (i, j) != (0, 0):
        # update the indexs
        # check the replace first
        if trace_matrix[i, j][2]:
            # replacing is true
            i, j = i-1, j-1
        elif trace_matrix[i, j][0]:
            # inserting is true
            i, j = i, j-1
        elif trace_matrix[i, j][1]:
            # deleting is true
            i, j = i-1, j
        backtrace_indexs.append((i, j))
    
    return backtrace_indexs
        

## visualize the process of dynamic programming
def _align_words(source, target, trace_indexs, verbose=False):
    """
    Input:
        source word
        target word
        trace_indexs: a list of trace indexs from trace matrix
    Output: aligned words
    """
    source_align = []
    target_align = []
    operations = []
    
    forward_trace = trace_indexs[::-1]  # reverse the backward trace indexs
    
    for k in range(len(forward_trace)-1):
        # get the index, 
        s_0, t_0 = forward_trace[k]
        s_1, t_1 = forward_trace[k+1]
        # initialize the letter
        s_letter = None
        t_letter = None
        op = None
        
        if s_1 > s_0 and t_1 > t_0:
            # either replacing or no operation
            if source[s_0] == target[t_0]:
                s_letter = source[s_0]
                t_letter = target[t_0]
                op = " "
            else:
                s_letter = source[s_0]
                t_letter = target[t_0]
                op = "r"  # replace
        elif s_0 == s_1:
            # row stays the same and it is inserting
            s_letter = " "  # keep the source letter the same and insert one
            t_letter = target[t_0]
            op = "i"
        else:
            # deleting
            s_letter = source[s_0]
            t_letter = " "
            op = "d"
        source_align.append(s_letter)
        target_align.append(t_letter)
        operations.append(op)
        
    if verbose:
        print(source_align, target_align, operations, sep='\n')
    
    return source_align, target_align, operations
                
    
def _make_table(source, target, cost_matrix, trace_matrix, trace_indexs):
    """
    print out a pretty table
    """
    m = len(source)
    n = len(target)
    table = np.zeros(cost_matrix.shape, dtype=object)
    
    for i in range(m+1):
        for j in range(n+1):
            # dtype=[('ins', 'b'), ('del', 'b'), ('rep', 'b')
            ic, dc, rc = trace_matrix[i, j]
            direction = (('\u2190' if ic else "") + ('\u2191' if dc else "") +
                         ('\u2196' if rc else ""))
            table[i, j] = direction + str(cost_matrix[i, j])
            if (i, j) in trace_index:
                table[i, j] = direction + str(cost_matrix[i, j]) + "*"
    
    idx = list('#' + source)
    cols = list('#' + target)
    df = pd.DataFrame(table, index=idx, columns= cols)
    
    print(df)
            


if __name__ == "__main__":
    source = 'intention'  # spell
    target = 'execution'  # hello
    min_edits, cost_matrix, trace = wagner_fischer(source, target)
    print("minimum edits: ", min_edits)
    print('cost matrix: \n', cost_matrix)
    print('-------------------------')
    print('trace matrix: \n', trace)
    print('-------------------------')
    trace_index = naive_trace(trace)
    print(trace_index)
    print('-------------------------')
    _align_words(source, target, trace_index, True)
    print('-------------------------')
    _make_table(source, target, cost_matrix, trace, trace_index)
  
    
    
    