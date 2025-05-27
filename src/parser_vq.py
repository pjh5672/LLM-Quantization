import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1)
        )
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.decoder.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    print(f"non-zeros: {total_params-count}, total: {total_params}")
    print(f"model sparsity {float(count)/total_params:.6f}")

def parser_vq_indices(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    # groupsize = 256 # w2d1
    groupsize = 512 # w2d1
    # groupsize = 1024 # w3d1
    columns_per_group = 256
    rows_per_group = groupsize // columns_per_group
    vq_dim = 1
    
    vq_encodings = {}
    layers = model.model.decoder.layers
    for i in tqdm(range(len(layers)), desc="Parsing VQ encodings..."):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W.shape
            groups_per_column = W.shape[0] // rows_per_group

            layer_masks = []
            layer_indices = []
            # layer_centroids = []
            for j in range(0, W.shape[1], columns_per_group):
                x = W[:, j:j+columns_per_group]
                x_reshaped = x.reshape(groups_per_column, -1, vq_dim)  # G x N x D

                masks = []
                # centroids = []
                assignments = []
                for k in range(x_reshaped.shape[0]): # For G
                    x_ = x_reshaped[k] # N x D
                    m_ = x_ == 0 # N x D
                    x_ = x_.unsqueeze(1) # N x 1 x D
                    c_ = torch.unique(x_).unsqueeze(1) # K x D
                    zero_index = (c_[:, 0] == 0).nonzero()
                    c_ = c_.unsqueeze(0) # 1 x K x D
                    dist = ((x_ - c_).pow(2)).sum(-1) # N x K
                    index = dist.argmin(-1) # N
                    ####### for sanity check #######
                    # values = torch.gather(c_, dim=1, index=index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, vq_dim))
                    # print(x_.squeeze(1))
                    # print(values[0])
                    # raise
                    ################################
                    # index re-sort
                    if len(zero_index):
                        for z in range(zero_index.item()+1, c_.shape[1]):
                            index[index == z] = z-1
                    masks.append(m_)
                    # centroids.append(c_)
                    assignments.append(index)
                masks = torch.stack(masks, dim=0) # G x N x D
                # centroids = torch.concat(centroids, dim=0) # G x K x D
                assignments = torch.stack(assignments, dim=0) # G x N
                layer_masks.append(masks)
                layer_indices.append(assignments)
                # layer_centroids.append(centroids)
            layer_masks = torch.concat(layer_masks, dim=1)  
            # layer_centroids = torch.stack(layer_centroids, dim=-1)         
            layer_indices = torch.concat(layer_indices, dim=1)

            vq_encodings[f"layer.{i}.{name}"] = {"sparsity_mask": layer_masks, "indices": layer_indices}
        
    model.config.use_cache = use_cache 
    return vq_encodings


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "./models/RIA-opt-125M-pr50-spp-sp-w2d1"
    # model_path = './models/opt-125M-w2d1'
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="eager", 
        torch_dtype=torch.float16, device_map="auto"
    )
    model.seqlen = 512
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # check_sparsity(model)
    vq_encodings = parser_vq_indices(model)
    print(vq_encodings)


