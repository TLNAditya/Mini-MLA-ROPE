import torch 

def precompute_freqs_cis(d_k,max_seq_len,base = 10000):
    thetha = 1/(base**(torch.arange(0,d_k,2).float()/d_k))
    positions = torch.arange(max_seq_len)
    freq = torch.outer(positions,thetha)
    freq = torch.polar(torch.ones_like(freq),freq)
    return freq


def apply_rotary_emb(q,k,freq_com):
    '''
    args:
        q shape = [B,seq_len,n_heads,d_k]
        k shape = [B,seq_len,n_heads,d_k]
        freq_com shape = [max_seq_len,d_k//2]
    return:
        q,k shape = [B,seq_len,n_heads,d_k]

    '''
    #this converts Q shape= [B, seq_len, n_heads, d_k//2,2] fpr ex: dk = 64 then-> [b,seq_len,n_heads,32,2]
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1],-1,2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1],-1,2))

    #we need to select top seq_len from pre computed complex table from precompute function
    freq_com = freq_com[:q_.shape[1]].unsqueeze(0).unsqueeze(2) #this converts freq com shape from (max_Seq_len,dk//2) -> shape(B=1,seq_len,n_heads=1,d_k//2)
    
    q_out = torch.view_as_real(q_*freq_com).flatten(3) 
    k_out = torch.view_as_real(k_*freq_com).flatten(3)
    '''
    q_*freq_com = complex multiplaction => it rotates every single vector by thetha
    torch.view_as_complex(..) -> adds additional dimensionality to the shape so the 
            q_out shape before doing this => Shape (B,seq_len,n_heads,d_k)
            q_out shape after doing this => Shape(B,seq_len,n_heads,d_k,2)
    So after applying flatten(3) to q_out => q_out Shape becomes (B,seq_len,n_head,d_k)
    How?
        for instance q_out before flatten, lets say (2,10,4,32,2) = (B,seq_len,n_heads,d_k)
        so flatten(3) starts at 3rd index of Shape and combines it which becomes 
            q_out Shape( 2, 10, 4, 64) => the original shape :)
    similarly for K vector
    '''
    return q_out.type_as(q),k_out.type_as(k)
    #this ensures the ouput is of same dtype as input


if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    num_heads = 4
    head_dim = 64  # Must be even
    
    # Dummy query and key tensors (normally outputs of W_q, W_k)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    # Precompute frequencies for max sequence length
    max_seq_len = 1024
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)
    
    # Apply RoPE
    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
    
    print("Input shapes:", q.shape, k.shape)
    print("Output shapes:", q_rot.shape, k_rot.shape)
    # Continue with scaled dot-product attention: q_rot @ k_rot.transpose(-2,-1)