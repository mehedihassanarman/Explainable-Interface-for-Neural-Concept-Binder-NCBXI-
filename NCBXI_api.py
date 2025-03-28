import os
import torch
import pickle
import numpy as np
from PIL import Image
from matplotlib.image import imread
from torchvision import transforms
import matplotlib.pyplot as plt
from NeuralConceptBinder.neural_concept_binder import NeuralConceptBinder
import matplotlib
matplotlib.use('Agg')


# Loads all parameters for model loading and evaluation
class Args:
    def __init__(self,model_path):
        # Generic Parameters
        self.name = "test_run"
        self.mode = "test"
        self.resume = None
        self.seed = 10
        self.epochs = 10
        self.lr = 1e-2
        self.l2_grads = 1.0
        self.batch_size = 32
        self.num_workers = 4
        self.no_cuda = False
        self.train_only = False
        self.eval_only = True
        self.multi_gpu = False
        self.data_dir = "../data/CLEVR-4-1/"
        self.model_path = model_path
        self.fp_ckpt = f"{self.model_path}best_model.pt"
        self.fp_pretrained_ckpt = None
        self.precompute_bind = False

        # SysBinder Arguments
        self.image_size = 128
        self.image_channels = 3
        self.lr_dvae = 3e-4
        self.lr_enc = 1e-4
        self.lr_dec = 3e-4
        self.lr_warmup_steps = 30000
        self.lr_half_life = 250000
        self.clip = 0.05
        self.num_iterations = 3
        self.num_slots = 1
        self.num_blocks = 16
        self.cnn_hidden_size = 512
        self.slot_size = 2048
        self.mlp_hidden_size = 192
        self.num_prototypes = 64
        self.temp = 1.0
        self.temp_step = False
        self.vocab_size = 4096
        self.num_decoder_layers = 8
        self.num_decoder_heads = 4
        self.d_model = 192
        self.dropout = 0.1
        self.tau_start = 1.0
        self.tau_final = 0.1
        self.tau_steps = 30000
        self.binarize = False
        self.attention_codes = False

        # Retrieval & Binding Arguments
        self.checkpoint_path = f"{self.model_path}best_model.pt"
        self.retrieval_corpus_path = f"{self.model_path}block_concept_dicts.pkl"
        self.retrieval_encs = "proto-exem"
        self.majority_vote = False
        self.topk = 5
        self.thresh_attn_obj_slots = 0.98
        self.thresh_count_obj_slots = -1
        self.deletion_dict_path = None
        self.merge_dict_path = None
        self.feedback_path = None
        self.expl_thresh = 0.5
        self.lambda_expl = 100
        self.set_transf_hidden = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


# Loads the pretrained NCB model
def load_model(args):
    model = NeuralConceptBinder(args)
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(args.device)
    model.eval()
    return model


# Preprocesses an image and returns tensor + original PIL image
def preprocess_image(image_path, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image


# Runs model inference and returns discrete concept codes and probabilities
def run_inference(image_tensor, model, device):
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        codes, probs = model.encode(image_tensor)
    return codes, probs


# Preprocesses paths for inspection functions
def preprocess_image_paths(all_img_locs):
    for i in range(len(all_img_locs)):
        all_img_locs[i] = "data/" + "/".join(all_img_locs[i].split("/")[3:])
    return all_img_locs


# Loads retrieval corpus of learned concepts and organizes them by block
def create_block_concepts(retrieval_corpus_path):
    print("Loading retrieval corpus...")
    with open(retrieval_corpus_path, "rb") as f:
        retrieval_corpus = pickle.load(f)

    block_concepts = {}
    for block_idx, block in enumerate(retrieval_corpus):
        if 'exemplars' in block:
            block_concepts[block_idx] = {
                'exemplars': block['exemplars'],
                'prototypes': block['prototypes']
            }
        else:
            print(f"Block {block_idx} has no exemplars.")

    print(f"Loaded block concepts: {list(block_concepts.keys())}")
    return block_concepts


# Converts tensor image to NumPy array for plotting
def tensor_img_to_np(tensor_img: torch.Tensor):
    tensor_img = tensor_img.detach().cpu()
    if len(tensor_img.shape) == 4:  # Batch of images
        img = tensor_img.permute(0, 2, 3, 1).numpy()
    else:  
        img = tensor_img.permute(1, 2, 0).numpy() # Single image
    return np.clip(img, 0, 1) 


# Reshapes slot encodings into blocks for individual concept inspection
def slots_to_blocks(slots, args):
    B, num_slots, slot_size = slots.shape
    block_size = slot_size // args.num_blocks
    assert slot_size % args.num_blocks == 0, "slot_size must be divisible by num_blocks"
    return slots.view(B, args.num_blocks, block_size)


# Loads an image as tensor for encoding (no PIL image returned)
def load_img_as_tensor(file_path: str, device: str):   
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    img = Image.open(file_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


# Identifies the slot most activated by computing distance between encodings. Infer the slot where the object is located based on encoding.
def slot_based_on_encoding(encoding: torch.tensor):
    encoding = encoding[0]
    res = np.zeros(encoding.shape[0])
    for i in range(encoding.shape[0]):
        res[i] = sum(torch.abs(encoding[i] - encoding[j]).sum().item() for j in range(encoding.shape[0]))
    return np.argmax(res)


# Visualizes the concept image that was activated in a specific block based on the encoding (concept ID) returned by the model.
def visualize_block(block_idx, codes,model_path):
    codes = codes.squeeze(0).cpu().numpy()
    base_path = f"{model_path}clustered_exemplars/"
    
    # Location for the output image of concept block for app.py
    output_plot_path = "static/images/plots/Visualize_Concept_Block/Concept_Block.png"

    concept_id = int(codes[0, block_idx])
    concept_image_path = os.path.join(base_path, f"block{block_idx}_{concept_id}.png")

    plt.figure(figsize=(8, 8))
    if os.path.exists(concept_image_path):
        concept_image = imread(concept_image_path)
        plt.imshow(concept_image)
        plt.title(f"Block {block_idx}, Concept {concept_id}", fontsize=14)
    else:
        plt.text(
            0.5, 0.5, "Image Not Found", fontsize=12, ha='center', va='center'
        )
        plt.title(f"Block {block_idx}, Concept {concept_id} (Image Not Found)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    # To make sure the directory exists
    os.makedirs(os.path.dirname(output_plot_path ), exist_ok=True)

    # Delete if there any previous image exists
    if os.path.exists(output_plot_path):
        os.remove(output_plot_path )

    # Save the output image
    plt.savefig(output_plot_path , bbox_inches="tight", dpi=600)
    plt.show()
    print(f'Plot of Visual Concept Block has been saved to "{output_plot_path}"')




# Displays example images (exemplars) for a specific concept cluster to understand what kind of visual patterns the concept represents.
def implicit_inspection(block_concepts, all_img_locs, block_id: int, cluster_id: int, exp_per_cluster: int = 5):

    # To raise error when block_id or cluster_id out of range
#    if block_id not in block_concepts:
#        raise ValueError(f"Block ID {block_id} is out of range. Available blocks: {list(block_concepts.keys())}.")
#    
#    if cluster_id >= len(block_concepts[block_id]['prototypes']['ids']):
#        raise ValueError(f"Cluster ID {cluster_id} is out of range for Block {block_id}. "
#                         f"Available clusters: {len(block_concepts[block_id]['prototypes']['ids']) - 1}.")
        
    # To print error instead of raise error when block_id or cluster_id out of range
    if block_id not in block_concepts:
        print(f"Block ID {block_id} is out of range. Available blocks: {list(block_concepts.keys())}.")
        return
    
    if cluster_id >= len(block_concepts[block_id]['prototypes']['ids']):
        print(f"Cluster ID {cluster_id} is out of range for Block {block_id}. "
              f"Available clusters: {len(block_concepts[block_id]['prototypes']['ids']) - 1}.")
        return
   
    # Location for the output image of Implicit Inspection for app.py
    output_plot_path = "static/images/plots/Implicit_Inspection/Implicit_Inspection.png"

    exemplar_ids = block_concepts[block_id]['exemplars']['exemplar_ids'][cluster_id]
    num_exemplars = len(exemplar_ids)

    if num_exemplars < exp_per_cluster:
        print(f"Cluster {cluster_id} in Block {block_id} has only {num_exemplars} exemplars. Adjusting.")
        exp_per_cluster = num_exemplars

    fig, axs = plt.subplots(1, exp_per_cluster, figsize=(4 * exp_per_cluster, 4))

    for i in range(exp_per_cluster):
        try:
            image_path = all_img_locs[exemplar_ids[i]]
            axs[i].imshow(imread("../" + image_path))  
            axs[i].axis("off")
            axs[i].set_title(f"Exemplar {i+1}")
        except IndexError:
            print(f"Skipped index {exemplar_ids[i]} (out of bounds in all_img_locs)")
            continue


    plt.suptitle(f"Implicit Inspection: Block {block_id}, Cluster {cluster_id}", fontsize=16)
    plt.tight_layout()

    # To make sure the directory exists
    os.makedirs(os.path.dirname(output_plot_path ), exist_ok=True)

    # Delete if there any previous image exists
    if os.path.exists(output_plot_path):
        os.remove(output_plot_path )

    # Save the output image
    plt.savefig(output_plot_path , bbox_inches="tight", dpi=300)
    plt.show()
    print(f'Plot of Implicit Inspection has been saved to "{output_plot_path}"')



# Compares an image’s activation with two concept clusters (closest vs specified) to highlight the differences in representation for comparative analysis.
def comparative_inspection(block_concepts, all_img_locs, model, example_path: str, block_id: int, cluster_id: int, num_exemplars: int = 3):

    # To raise error when block_id or cluster_id out of range
#    if block_id not in block_concepts:
#        raise ValueError(f"Block ID {block_id} is out of range. Available blocks: {list(block_concepts.keys())}.")
    
#    if cluster_id >= len(block_concepts[block_id]['prototypes']['ids']):
#        raise ValueError(f"Cluster ID {cluster_id} is out of range for Block {block_id}. "
#                         f"Available clusters: {len(block_concepts[block_id]['prototypes']['ids']) - 1}.")

    # To print error instead of raise error when block_id or cluster_id out of range
    if block_id not in block_concepts:
        print(f"Block ID {block_id} is out of range. Available blocks: {list(block_concepts.keys())}.")
        return
    
    if cluster_id >= len(block_concepts[block_id]['prototypes']['ids']):
        print(f"Cluster {cluster_id} is out of range for Block {block_id}. "
              f"Available clusters: {len(block_concepts[block_id]['prototypes']['ids']) - 1}.")
        return

    # Location for the output image of Comparative Inspection for app.py
    output_plot_path = "static/images/plots/Comparative_Inspection/Comparative_Inspection.png"

    # Encode input image
    example_tensor, _ = preprocess_image(example_path, 128)
    example_tensor = example_tensor.to(model.device)
    enc, _ = run_inference(example_tensor, model, model.device)

    # Identify closest cluster from encoding
    if len(enc.shape) == 2:
        closest_cluster = int(enc[0, block_id].item())
    elif len(enc.shape) == 3:
        slot = slot_based_on_encoding(enc)
        closest_cluster = int(enc[0, slot, block_id].item())
    else:
        print(f"Unexpected encoding shape: {enc.shape}")
        return

    print(f"Closest cluster for Block {block_id}: {closest_cluster}. Comparing against Cluster {cluster_id}.")

    # Fetch exemplar IDs
    exemplars_closest = list(block_concepts[block_id]['exemplars']['exemplar_ids'][closest_cluster][:num_exemplars])
    exemplars_different = list(block_concepts[block_id]['exemplars']['exemplar_ids'][cluster_id][:num_exemplars])

    # Warn if not enough
    if len(exemplars_closest) < num_exemplars:
        print(f"Closest cluster {closest_cluster} has only {len(exemplars_closest)} exemplars.")
    if len(exemplars_different) < num_exemplars:
        print(f"Cluster {cluster_id} has only {len(exemplars_different)} exemplars.")

    all_exemplars = exemplars_closest + exemplars_different
    while len(all_exemplars) < num_exemplars * 2:
        all_exemplars.append(all_exemplars[-1])

    fig, axs = plt.subplots(1, 1 + num_exemplars * 2, figsize=(5 * (1 + num_exemplars * 2), 5))

    # Show input image
    axs[0].imshow(imread(example_path))
    axs[0].axis("off")
    axs[0].set_title("Input Image", fontsize=12)

    # Show exemplars
    for i in range(num_exemplars * 2):
        try:
            img_path = "../" + all_img_locs[all_exemplars[i]]
            axs[i + 1].imshow(imread(img_path))
            axs[i + 1].axis("off")
            label = "Closest" if i < num_exemplars else "Different"
            axs[i + 1].set_title(f"{label} {i % num_exemplars + 1}", fontsize=10)
        except IndexError:
            print(f"Skipped invalid exemplar index: {all_exemplars[i]}")

    plt.suptitle(f"Comparative Inspection: Block {block_id}\nClosest Cluster {closest_cluster} vs. Cluster {cluster_id}", fontsize=14, y=1.05)
    plt.tight_layout()

    # To make sure the directory exists
    os.makedirs(os.path.dirname(output_plot_path ), exist_ok=True)

    # Delete if there any previous image exists
    if os.path.exists(output_plot_path):
        os.remove(output_plot_path )

    # Save the output image
    plt.savefig(output_plot_path , bbox_inches="tight", dpi=600)
    plt.show()
    print(f'Plot of Comparative Inspection has been saved to "{output_plot_path}"')



# Intervenes in the model's latent space by swapping a block with a prototype concept to observe the effect of concept-level editing on the output image.
def interventional_inspection(block_concepts, model, example_path: str, block_id: int, cluster_id: int, args):

    # To raise error when block_id or cluster_id out of range
#    if block_id not in block_concepts:
#        raise ValueError(f"Block ID {block_id} is out of range. Available blocks: {list(block_concepts.keys())}.")
    
#    if cluster_id >= len(block_concepts[block_id]['prototypes']['ids']):
#        raise ValueError(f"Cluster ID {cluster_id} is out of range for Block {block_id}. "
#                         f"Available clusters: {len(block_concepts[block_id]['prototypes']['ids']) - 1}.")
    

    # To print error instead of raise error when block_id or cluster_id out of range
    if block_id not in block_concepts:
        print(f"Block ID {block_id} is out of range. Available blocks: {list(block_concepts.keys())}.")
        return
    
    if cluster_id >= len(block_concepts[block_id]['prototypes']['ids']):
        print(f"Cluster ID {cluster_id} is out of range for Block {block_id}. "
              f"Available clusters: {len(block_concepts[block_id]['prototypes']['ids']) - 1}.")
        return
    
    # Location for the output image of Interventional Inspection for app.py
    output_plot_path = "static/images/plots/Interventional_Inspection/Interventional_Inspection.png"

    # Load the image as a tensor
    img_tensor = load_img_as_tensor(example_path, model.device)

    # Encode the image
    enc = model.model.encode(img_tensor)[0]

    # Convert slots to blocks
    enc = slots_to_blocks(enc, args)
    #print(f"Encoding shape after reshaping to blocks: {enc.shape}")

    # Verify block_id is valid
    num_blocks = enc.size(2)  
    #print(f"Available blocks: {num_blocks}")
    assert block_id < num_blocks, f"Block ID {block_id} is out of bounds. Available blocks: {num_blocks - 1}"

    # Retrieve the prototype encoding
    proto_enc = torch.tensor(
        block_concepts[block_id]['prototypes']['prototypes'][cluster_id],
        device=enc.device
    ).unsqueeze(0)  

    # Resize prototype if necessary
    if proto_enc.shape[-1] != enc.shape[-1]:
        print(f"Resizing prototype from {proto_enc.shape[-1]} to {enc.shape[-1]}")
        proto_enc = torch.nn.functional.pad(proto_enc, (0, enc.shape[-1] - proto_enc.shape[-1]))

    # Validate prototype dimensions
    assert proto_enc.shape[-1] == enc.shape[-1], (
        f"Prototype tensor size mismatch: expected {enc.shape[-1]}, got {proto_enc.shape[-1]}"
    )

    # Swap the block encoding with the prototype
    swapped_enc = enc.clone()
    swapped_enc[:, block_id, :] = proto_enc

    # Reshape back to original slot dimensions for decoding
    swapped_slots = swapped_enc.view(enc.size(0), -1, args.slot_size)

    # Decode the swapped encoding
    swapped_img = tensor_img_to_np(model.model.decode(swapped_slots))[0]

    # Display the swapped image with the specified title
    plt.figure(figsize=(4, 4))
    plt.imshow(swapped_img)
    plt.axis('off')
    plt.title(f"Interventional Inspection: Block {block_id}, Cluster {cluster_id}", fontsize=11)
    plt.tight_layout()

    # To make sure the directory exists
    os.makedirs(os.path.dirname(output_plot_path ), exist_ok=True)

    # Delete if there any previous image exists
    if os.path.exists(output_plot_path):
        os.remove(output_plot_path )

    # Save the output image
    plt.savefig(output_plot_path , bbox_inches="tight", dpi=600)
    plt.show()
    print(f'Plot of Interventional Inspection has been saved to "{output_plot_path}"')