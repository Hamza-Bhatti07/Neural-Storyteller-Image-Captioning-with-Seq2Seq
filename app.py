import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import pickle
import numpy as np
import random # Added for consistency with your snippet

# ==========================================
# 1. CONFIGURATION
# ==========================================
EMBED_DIM = 256
HIDDEN_SIZE = 512
MAX_LEN = 30
device = torch.device("cpu") # Hugging Face Free Tier uses CPU

# Special Tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"

# ==========================================
# 2. LOAD VOCABULARY
# ==========================================
with open('vocab.pkl', 'rb') as f:
    idx2word = pickle.load(f)

word2idx = {v: k for k, v in idx2word.items()}

PAD_ID = word2idx.get(PAD_TOKEN, 0)
START_ID = word2idx.get(START_TOKEN, 1)
END_ID = word2idx.get(END_TOKEN, 2)
UNK_ID = word2idx.get(UNK_TOKEN, 3)
VOCAB_SIZE = len(word2idx)

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class Encoder(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.fc = nn.Linear(2048, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()

    def forward(self, img_feat):
        x = self.dropout(self.activation(self.fc(img_feat)))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, cap_input, hidden_state, cell_state):
        embeddings = self.dropout(self.embed(cap_input))
        output, (h_n, c_n) = self.lstm(embeddings, (hidden_state, cell_state))
        logits = self.fc(output)
        return logits, (h_n, c_n)

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size)

    def forward(self, img_feat, cap_input):
        img_embed = self.encoder(img_feat)
        h_0 = img_embed.unsqueeze(0)
        c_0 = img_embed.unsqueeze(0)
        logits, _ = self.decoder(cap_input, h_0, c_0)
        return logits

# Load Model Weights
model = Seq2Seq(VOCAB_SIZE).to(device)
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.eval()

# Preprocessing transforms
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ==========================================
# 4. BEAM SEARCH LOGIC (Required for inference)
# ==========================================
def beam_search(img_feat_tensor, beam_width=5, max_len=MAX_LEN):
    model.eval()
    
    # Encode & Init
    img_embed = model.encoder(img_feat_tensor.unsqueeze(0).to(device))
    h = img_embed.unsqueeze(0)
    c = img_embed.unsqueeze(0)
    
    beams = [([START_ID], 0.0, h, c)]
    
    for _ in range(max_len):
        new_beams = []
        for seq, score, h_curr, c_curr in beams:
            if seq[-1] == END_ID:
                new_beams.append((seq, score, h_curr, c_curr))
                continue
            
            inp = torch.tensor([[seq[-1]]], device=device)
            embed = model.decoder.embed(inp)
            out, (h_next, c_next) = model.decoder.lstm(embed, (h_curr, c_curr))
            logits = model.decoder.fc(out) 
            log_probs = torch.log_softmax(logits[0, 0], dim=0)
            
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
            
            for k in range(beam_width):
                next_id = topk_ids[k].item()
                if len(seq) > 1 and next_id == seq[-1]: continue # Block repetition
                new_score = score + topk_log_probs[k].item()
                new_beams.append((seq + [next_id], new_score, h_next, c_next))
        
        beams = sorted(new_beams, key=lambda x: x[1] / (len(x[0])**0.7), reverse=True)[:beam_width]
        if all(b[0][-1] == END_ID for b in beams): break

    best_seq = beams[0][0]
    tokens = [idx2word.get(i, UNK_TOKEN) for i in best_seq if i not in {START_ID, END_ID, PAD_ID}]
    return " ".join(tokens)

# =============================================================================
# 7. APP DEPLOYMENT (GRADIO) - YOUR REQUESTED UI
# =============================================================================

# Load ResNet for live inference (independent of cached features)
resnet_live = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet_live = nn.Sequential(*list(resnet_live.children())[:-1])
resnet_live = resnet_live.to(device)
resnet_live.eval()

def generate_caption_app(image):
    """
    Inference function for Gradio.
    Takes a PIL image, extracts features, and generates a caption.
    """
    if image is None:
        return "Please upload an image."
    
    try:
        # Preprocess image
        img_tensor = resnet_transform(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            feature = resnet_live(img_tensor).view(1, -1) # (1, 2048)
            feature = feature.squeeze(0) # (2048,)
        
        # Generate Caption
        caption = beam_search(feature, beam_width=5)
        return caption.capitalize()
    except Exception as e:
        return f"Error: {str(e)}"

# Define Gradio Interface
interface = gr.Interface(
    fn=generate_caption_app,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Neural Storyteller",
    description="Upload an image to generate a descriptive caption using the ResNet50 + Seq2Seq model.",
    theme="default"
)

# Launch the app
print("Launching Gradio App...")
interface.launch()