import typing as T
from math import inf

import torch
from torch.nn.functional import pad
from torch import Tensor
import einops



def find_cycle(par: Tensor, root: int, n) -> list:
    seen = torch.full((n,), -1, dtype=torch.long, device=par.device)
    # print("seen:", seen)
    cycle = []
    vid = 0
    for i in range(n):
        # print("i:", i )
        if i == root:
            continue
        if seen[i] != -1:
            continue
        cur = i
        path_index = {}
        while cur != root and seen[cur] == -1:
            path_index[cur] = len(path_index)
            # print("path_index[cur]:", path_index[cur])
            seen[cur] = vid
            # print("seen[cur]:", seen[cur])
            cur = par[cur].item()
            if cur in path_index:  # found cycle
                start = cur

                cyc_nodes = [k for k, _ in sorted(path_index.items(), key=lambda x: x[1])]
                start_pos = path_index[start]
                cycle = cyc_nodes[start_pos:]
                return cycle
        vid += 1
    return cycle  # empty if none


def contract(n, C):
    # print("###"*40)
    # contract the cycle into a super-node
    # map old indices -> new indices
    new_index = {}
    rev_index = []
    c_idx = None
    cnt = 0
    # print("n:", n)
    for v in range(n):
        # print("V", v)
        # print(C)
        if v in C:
            # print("in C")
            if c_idx is None:
                # print("c_idx is None")
                # print("cnt:", cnt)  
                c_idx = cnt
                # print("before adding to new_index:", new_index)
                new_index[v] = c_idx
                # print("after adding to new_index:", new_index)
                rev_index.append(v)  # representative for cycle
                # print("after adding to rev_index:", rev_index)
                cnt += 1
                # print("cnt:", cnt)
            else:
                # print("c_idx is not None")
                new_index[v] = c_idx 
        else:
            # print("not in C")
            # print("new_index:", new_index)
            new_index[v] = cnt
            # print("new_index:", new_index)
            rev_index.append(v)
            # print("rev_index:", rev_index)
            cnt += 1
            # print("cnt:", cnt)

    return cnt, new_index, c_idx, rev_index


def expand(par, C, n, par_new, new_index, c_idx, scores, rev_index):
    # Expand
    par_exp = par.clone()

    for i in range(n):
        if i in C:
            continue
        pi_new = par_new[new_index[i]].item()
        if pi_new == c_idx:
            # (max over scores[i, v])
            best_v = None
            best_val = float('-inf')
            for v in C:
                val = scores[i, v].item()
                if val > best_val:
                    best_val = val
                    best_v = v
            par_exp[i] = best_v
        else:
            # head maps back directly
            head_old = rev_index[pi_new]
            par_exp[i] = head_old

    return par_exp


def mstSingleRoot(arcTensor: Tensor, lengths: Tensor) -> Tensor:
    """
    Finds the maximum spanning tree (more technically, arborescence) for the
    given sentences such that each tree has a single root word.

    Remember that index 0 indicates the ROOT node. A tree with "a single root
    word" has exactly one outgoing edge from ROOT.

    If you like, you may add helper functions to this file for this function.

    This file already imports the function `pad` for you. You may find that
    function handy. Here's the documentation of the function:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

    Args:
        arcTensor (Tensor): a Tensor of dimensions (batch_sz, x, y) and dtype
            float where x=y and the entry at index (b, i, j) indicates the
            score for a candidate arc from vertex j to vertex i.

        lengths (Tensor): a Tensor of dimensions (batch_sz,) and dtype int
            where each element indicates the number of words (this doesn't
            include ROOT) in the corresponding sentence.

    Returns:
        A Tensor of dtype int and dimensions (batch_sz, x) where the value at
        index (b, i) indicates the head for vertex i according to the
        maximum spanning tree for the input graph.

    Examples:
        >>> mstSingleRoot(torch.tensor(\
            [[[0, 0, 0, 0],\
              [12, 0, 6, 5],\
              [4, 5, 0, 7],\
              [4, 7, 8, 0]],\
             [[0, 0, 0, 0],\
              [1.5, 0, 4, 0],\
              [2, 0.1, 0, 0],\
              [0, 0, 0, 0]],\
             [[0, 0, 0, 0],\
              [4, 0, 3, 1],\
              [6, 2, 0, 1],\
              [1, 1, 8, 0]]]),\
            torch.tensor([3, 2, 3]))
        tensor([[0, 0, 3, 1],
                [0, 2, 0, 0],
                [0, 2, 0, 2]])
    """
    def chu_liu_edmonds(scores: Tensor, root: int = 0) -> Tensor:
        n = scores.shape[0]

        # print("n:", n)

        # pick best incoming head for each node 
        par = torch.full((n,), -1, dtype=torch.long, device=scores.device)
        par[root] = root
        # print("par init:", par)
        # exit()

        S = scores.clone() # original scores
        # print("S1:", S)
        S[root, :] = float('-inf')  # root has no incoming head
        # print("S2:", S)
        S[root, root] = 0.0
        # print("S3:", S)
        idx = torch.arange(n, device=scores.device)
        # print("idx:", idx)
        S[idx, idx] = float('-inf')  # no self loops (root handled already)
        # print("S4:", S)
        S[root, root] = 0.0
        # print("S5:", S)

        # print("*"*40)

        # best incoming head for i != root
        best_heads = torch.argmax(S, dim=1)
        # print("best_heads:", best_heads)

        # exit()
        # print("root:", root)

        # assign best head to par
        for i in range(n):
            # print("&&&&&&&&&&&&&&&&&&&7")
            # print("i", i)
            # print("best_heads[i]:", best_heads[i])
            # print("par[i]:", par[i])
            if i == root:
                # print("in if ")
                continue
            par[i] = best_heads[i].item()

        # print("par:", par)
        # exit()

        # check for cycle
        cycle = find_cycle(par, root, n)
        # print("%"*40)
        # print("cycle:", cycle)
        # exit()
        
        if not cycle:
            # print("no cycle")
            # exit()
            return par  # no cycles 


        # cyle found
        C = set(cycle)
        # print("C:", C)

        # contract
        cnt, new_index, c_idx, rev_index = contract(n, C)
        # print("cnt:", cnt)
        # print("new_index:", new_index)
        # # exit()
        # print("@"*20)
        # print('after contracting')
        # print("cnt and N:", cnt)
        N = cnt  # new size
        S_new = torch.full((N, N), float('-inf'), dtype=scores.dtype, device=scores.device)
        # print("S_new:", S_new)

        w_in = {v: scores[v, par[v]].item() for v in C}
        # print("w_in:", w_in)


        # copies entries from scores into a smaller matrix S_new, 
        # excluding all rows and columns whose indices are in C.
        for i in range(n):
            if i in C:
                continue
            for j in range(n):
                if j in C:
                    continue
                S_new[new_index[i], new_index[j]] = scores[i, j]
        # print("S_new:", S_new)
        
        # add edgee
        enter_arg = {}
        for a in range(n):
            if a in C:
                continue
            best_val = float('-inf')
            best_v = None
            for v in C:
                val = scores[v, a].item() - w_in[v]
                if val > best_val:
                    best_val = val
                    best_v = v
            S_new[c_idx, new_index[a]] = best_val
            enter_arg[a] = best_v

        # print("enter_arg:", enter_arg)

        # found the strongest 
        for i in range(n):
            if i in C:
                continue
            best_val = float('-inf')
            for v in C:
                val = scores[i, v].item()
                if val > best_val:
                    best_val = val
            S_new[new_index[i], c_idx] = best_val

        new_root = new_index[root]
        # print("new_root:", new_root)
        # print("S_new:", S_new)

        # Recurse on the contracted graph
        par_new = chu_liu_edmonds(S_new, root=new_root)
        

        # Expand
        par_exp = expand(par, C, n, par_new, new_index, c_idx, scores, rev_index)

        head_into_cycle_new = par_new[c_idx].item()
        head_into_cycle_old = rev_index[head_into_cycle_new]  # this is an old node outside or root
        v_star = enter_arg[head_into_cycle_old]

        par_exp[v_star] = head_into_cycle_old
        # print(par_exp)
        return par_exp

    B, X, Y = arcTensor.shape
    # print("B, X, Y", B, X, Y)

    result_heads = torch.zeros((B, X), dtype=torch.long, device=arcTensor.device)

    # print(result_heads)

    for b in range(B):
        # print('*'*20, 'in sentence ', b, '*'*20)
        L = int(lengths[b].item())
        # print("L", L)
        n = L + 1
        S = arcTensor[b, :n, :n].clone()
        # print("S", S)


        # Run cle
        heads = chu_liu_edmonds(S, root=0)
        # print("heads", heads)
        # exit()
        
        n = L + 1
        def count_root_children(h, n):
            # returns absolute node ids in 1..n-1 whose head is 0
            return (h[1:n] == 0).nonzero(as_tuple=False).add(1).flatten().tolist()
        
        root_children = count_root_children(heads, n)
        # print("root_children", root_children)
        # exit()


        guard = 0
        while len(root_children) > 1:
            # deal with multiple root children

            # print("Multiple root children, enforcing single-root...")
            scores_to_root = S[root_children, 0]
            # print("scores_to_root", scores_to_root)
            keep_idx = root_children[torch.argmax(scores_to_root).item()]
            # print("keep_idx (absolute node id):", keep_idx)

            for k in root_children:
                # print("k (absolute node id):", k)
                if k != keep_idx:
                    # print("k != keep_idx")
                    S[k, 0] = float('-inf')

            # re-run Edmonds
            heads = chu_liu_edmonds(S, root=0)
            # print("heads", heads)

            root_children = count_root_children(heads, n)
            # print("root_children", root_children)

            guard += 1
            if guard > n:  
                raise RuntimeError("single-root enforcement did not converge")


        # store into output (pad remaining positions with 0)
        result_heads[b, :n] = heads
        if n < X:
            result_heads[b, n:] = 0  # irrelevant/padded positions

        # print("result_heads", result_heads)
        # exit()

    # print("*"*20, "Done", "*" * 20)
    # print("result_heads", result_heads)


    return result_heads

input1 = torch.tensor(\
            [[[0, 0, 0, 0],\
              [12, 0, 6, 5],\
              [4, 5, 0, 7],\
              [4, 7, 8, 0]],\
             [[0, 0, 0, 0],\
              [1.5, 0, 4, 0],\
              [2, 0.1, 0, 0],\
              [0, 0, 0, 0]],\
             [[0, 0, 0, 0],\
              [4, 0, 3, 1],\
              [6, 2, 0, 1],\
              [1, 1, 8, 0]]])

input2 = torch.tensor([3, 2, 3])

print(mstSingleRoot(input1, input2))