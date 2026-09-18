import os
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm 
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# --- 1. 配置与常量 ---
SEQUENCE_LENGTH = 81
VOCAB_SIZE = 4        
EMBEDDING_DIM = 2560  # LucaOne的输出维度，经过最大池化后
LATENT_DIM = 64       # VAE潜在空间维度
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  
NUM_EPOCHS = 200
BETA = 1.0            
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_FILE_PATH = ""                         # 替换为您的CSV文件路径
EMBEDDING_DIR = ""                         # 替换为您的embedding文件夹路径
SEQUENCE_COLUMN_NAME = "seq"               # 替换为您的CSV中序列列的名称
MATRIX_FILE_PATTERN = "matrix_{}.pt"       # 替换为您的.pt文件的命名模式，{}会被索引替换
CHECKPOINT_DIR = ""                        # 保存检查点的目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Create directory if it doesn't exist
FINAL_MODEL_PATH = "final_optimal_LucaVAE_model.pth"

# --- 2. 数据集定义 ---
class PromoterDataset(Dataset):
    def __init__(self, csv_file_path, embeddings_dir_path, sequence_column_name, matrix_file_pattern, sequence_length=SEQUENCE_LENGTH):
        self.csv_file_path = csv_file_path
        self.embeddings_dir_path = embeddings_dir_path
        self.sequence_column_name = sequence_column_name
        self.matrix_file_pattern = matrix_file_pattern
        self.sequence_length = sequence_length

        print(f"Loading sequences from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        self.sequences = df[self.sequence_column_name].tolist()
        print(f"Found {len(self.sequences)} sequences.")

        # Validate that embedding files exist for all sequences
        missing_files = []
        for i in range(len(self.sequences)):
            emb_file = os.path.join(embeddings_dir_path, matrix_file_pattern.format(i + 1)) # Index starts from 1
            if not os.path.exists(emb_file):
                missing_files.append(emb_file)
        
        if missing_files:
            raise FileNotFoundError(f"Some embedding files are missing: {missing_files[:5]}... (showing first 5)")
        
        # Load all corresponding embedding tensors
        self.embeddings = []
        for i in range(len(self.sequences)):
            emb_file = os.path.join(embeddings_dir_path, matrix_file_pattern.format(i + 1))          # Index starts from 1
            emb_numpy_array = torch.load(emb_file, map_location='cpu', weights_only=False).squeeze() # Load numpy array
            emb_tensor = torch.from_numpy(emb_numpy_array).float()                                   # Convert numpy array to PyTorch tensor
            
            # --- Apply mean pooling across the sequence length dimension (dim=0) to get [EMBEDDING_DIM] ---
            pooled_emb_tensor = torch.mean(emb_tensor, dim=0) # [2560]
            
            if pooled_emb_tensor.shape[0] != EMBEDDING_DIM:
                 raise ValueError(f"Pooled embedding file {emb_file} has unexpected shape {pooled_emb_tensor.shape}, expected [{EMBEDDING_DIM}]")
            self.embeddings.append(pooled_emb_tensor) 
        
        assert len(self.sequences) == len(self.embeddings), "Number of sequences must match number of embeddings."
        print(f"Loaded and processed {len(self.embeddings)} embeddings.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_str = self.sequences[idx]
        embedding_tensor = self.embeddings[idx] 

        # Verify sequence length matches expectation
        if len(sequence_str) != self.sequence_length:
            print(f"Warning: Sequence at index {idx} has length {len(sequence_str)}, expected {self.sequence_length}. It will be truncated/padded.")

        # Convert sequence string to one-hot encoded tensor
        sequence_one_hot = self._seq_to_onehot(sequence_str)

        return sequence_one_hot, embedding_tensor

    def _seq_to_onehot(self, seq):
        """Convert a DNA sequence string to a one-hot encoded tensor."""
        # Pad or truncate sequence to desired length
        seq = seq.ljust(self.sequence_length, 'A')[:self.sequence_length]
        
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        indices = torch.tensor([mapping.get(nuc.upper(), 0) for nuc in seq], dtype=torch.long) # Default to 'A' if unknown
        # Use torch.nn.functional.one_hot and permute dimensions
        one_hot = torch.nn.functional.one_hot(indices, num_classes=VOCAB_SIZE).float()
        return one_hot # Shape: [SEQUENCE_LENGTH, VOCAB_SIZE]
    
# --- 3. VAE组件定义 ---
class Encoder(nn.Module):
    def __init__(self, sequence_length, vocab_size, embedding_dim_cond, latent_dim):
        super(Encoder, self).__init__()
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        input_dim = sequence_length * vocab_size # Flatten sequence

        # Process sequence
        self.fc_seq = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()

        # Combine sequence and condition features (condition not used in encoder for this version)
        combined_input_dim = 512 
        self.fc_mu = nn.Linear(combined_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_input_dim, latent_dim)

    def forward(self, x, c=None): # c is the condition (embedding), optional here but passed for consistency
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)     # Flatten sequence: [B, L*V]

        h1 = self.relu(self.fc_seq(x_flat)) # Process sequence: [B, 512]

        h_combined = h1 # [B, 512]

        mu = self.fc_mu(h_combined)         # [B, latent_dim]
        logvar = self.fc_logvar(h_combined) # [B, latent_dim]
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, sequence_length, vocab_size, embedding_dim_cond, latent_dim):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim_cond = embedding_dim_cond
        output_dim = sequence_length * vocab_size

        # Process combined latent variable z and condition c
        input_dim = latent_dim + embedding_dim_cond
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1) # Apply softmax along the vocab dimension

    def forward(self, z, c):
        # z: [B, latent_dim], c: [B, embedding_dim_cond] 
        z_c = torch.cat((z, c), dim=1) # Concatenate along feature dimension: [B, latent_dim + embedding_dim_cond]

        h1 = self.relu(self.fc1(z_c)) # [B, 512]
        out = self.fc2(h1)            # [B, L*V]
        out = out.view(-1, self.sequence_length, self.vocab_size) # Reshape to [B, L, V]
        out = self.softmax(out) # Apply Softmax over vocabulary dimension
        return out # Probabilities for each position


class LucaVAE(nn.Module):
    def __init__(self, sequence_length, vocab_size, embedding_dim_cond, latent_dim):
        super(LucaVAE, self).__init__()
        self.encoder = Encoder(sequence_length, vocab_size, embedding_dim_cond, latent_dim)
        self.decoder = Decoder(sequence_length, vocab_size, embedding_dim_cond, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)    # Pass c to encoder (though not used in this impl)
        z = self.reparameterize(mu, logvar)
        recon_x_probs = self.decoder(z, c) # Pass both z and c to decoder
        return recon_x_probs, mu, logvar

# --- 4. 损失函数定义 ---
def loss_function(recon_x_probs, x, mu, logvar, beta=BETA):
    # x is one-hot, recon_x_probs is probabilities from decoder's softmax
    log_probs = torch.log(recon_x_probs + 1e-8)        # Add small epsilon to avoid log(0)
    recon_loss = -torch.sum(x * log_probs) / x.size(0) # Average over batch

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss.item(), kl_loss.item()

# --- 5. 训练函数 ---
def train_model(model, dataloader, optimizer, device, epoch, best_val_loss=float('inf')):
    model.train()
    train_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    # Use tqdm for the epoch's batch loop
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    for batch_idx, (data, cond) in enumerate(pbar):
        data = data.to(device) # [B, L, V]
        cond = cond.to(device) # [B, E] where E=2560

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, cond)
        loss, recon_loss_val, kl_loss_val = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        recon_loss_total += recon_loss_val
        kl_loss_total += kl_loss_val
        optimizer.step()

        # Update tqdm description with current loss
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_loss = train_loss / len(dataloader)
    avg_recon_loss = recon_loss_total / len(dataloader)
    avg_kl_loss = kl_loss_total / len(dataloader)
    # print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL loss: {avg_kl_loss:.4f}')
    return avg_loss, avg_recon_loss, avg_kl_loss 

# --- 6. 生成函数 ---
def generate_sequence(model, cond_embedding, device, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # Ensure cond_embedding is on the right device and has batch dimension
        cond_tensor = cond_embedding.unsqueeze(0).to(device) # [1, E] where E=2560

        # Sample from prior (standard normal) in latent space
        z = torch.randn(1, LATENT_DIM, device=device)        # [1, Z]

        # Decode using the sampled z and the provided condition
        recon_probs = model.decoder(z, cond_tensor)          # [1, L, V]

        # Apply temperature scaling (affects sampling randomness)
        if temperature != 1.0:
            recon_probs = recon_probs / temperature
            recon_probs = torch.nn.functional.softmax(recon_probs, dim=-1) # Renormalize after scaling

        # Sample from the output probability distribution
        sampled_indices = torch.multinomial(recon_probs.squeeze(0), num_samples=1).squeeze(-1) # [L]

        # Convert indices back to sequence string
        idx_to_nuc = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        generated_seq = "".join([idx_to_nuc[i.item()] for i in sampled_indices])

        return generated_seq
    
def main():
    print(f"Using device: {DEVICE}")

    # Load dataset
    try:
        dataset = PromoterDataset(CSV_FILE_PATH, EMBEDDING_DIR, SEQUENCE_COLUMN_NAME, MATRIX_FILE_PATTERN)
    except FileNotFoundError as e:
        print(f"Error loading  {e}")
        return

    # Split dataset into train and val (simple split, you might want stratified or species-based splits)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False) # No shuffling or dropping for validation

    # Initialize model
    model = LucaVAE(
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        embedding_dim_cond=EMBEDDING_DIM, 
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Track best validation loss and save its state
    best_val_loss = float('inf')
    best_epoch = -1

    # --- Lists to store metrics for plotting ---
    epochs_list = []
    train_avg_losses = []
    train_recon_losses = []
    train_kl_losses = []
    val_avg_losses = []

    # Training loop
    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train one epoch and get losses
        train_avg_loss, train_recon_loss, train_kl_loss = train_model(model, train_dataloader, optimizer, DEVICE, epoch)
        
        # Append training metrics
        epochs_list.append(epoch)
        train_avg_losses.append(train_avg_loss)
        train_recon_losses.append(train_recon_loss)
        train_kl_losses.append(train_kl_loss)

        # Validation step (for finding optimal checkpoint)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, cond in val_dataloader:
                data = data.to(DEVICE)
                cond = cond.to(DEVICE) 
                recon_batch, mu, logvar = model(data, cond)
                loss, _, _ = loss_function(recon_batch, data, mu, logvar)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        val_avg_losses.append(avg_val_loss)
        # print(f'----> Epoch: {epoch} Average Val Loss: {avg_val_loss:.4f}')

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            # Save the model state dict, optimizer state, epoch, and best val loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(CHECKPOINT_DIR, f"best_checkpoint_epoch_{epoch}.pth.tar"))
            # print(f"------> New best model saved at epoch {epoch} with val loss {avg_val_loss:.4f}")

    # After training, save the final model state dict
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, FINAL_MODEL_PATH))
    print(f"\nTraining finished. Final model saved as '{os.path.join(CHECKPOINT_DIR, FINAL_MODEL_PATH)}'.")
    print(f"Best model was at epoch {best_epoch} with validation loss {best_val_loss:.4f}.")

    # --- Plotting the loss curves ---
    print("\nPlotting loss curves...")
    plt.figure(figsize=(15, 10)) # Slightly larger figure for 4 subplots

    # Subplot 1: Train Average Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_list, train_avg_losses, linestyle='-') 
    plt.title('Train Average Total Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6) 

    # Subplot 2: Train Average Reconstruction Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs_list, train_recon_losses, linestyle='-', color='orange') 
    plt.title('Train Average Reconstruction Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Subplot 3: Train Average KL Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs_list, train_kl_losses, linestyle='-', color='green') 
    plt.title('Train Average KL Divergence Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('KL Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Subplot 4: Validation Average Total Loss
    plt.subplot(2, 2, 4)
    plt.plot(epochs_list, val_avg_losses, linestyle='-', color='red') 
    plt.title('Validation Average Total Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(CHECKPOINT_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=300) # Higher DPI for better quality
    plt.show() 
    print(f"Training curves saved to: {plot_path}")

    # *** Load the BEST model for generation ***
    print(f"\nLoading the best model from epoch {best_epoch} for generation...")
    
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_checkpoint_epoch_{best_epoch}.pth.tar")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Optionally reload optimizer state if resuming training later
        print(f"Successfully loaded best model from {best_checkpoint_path}.")
    else:
        print(f"WARNING: Best checkpoint file {best_checkpoint_path} not found! Using current final model state.")

    
    print("\nGenerating new sequences using the optimal model...")

    # --- Configuration for generation ---
    NUM_GENERATIONS_PER_CONDITION = 5   # Set how many sequences you want to generate per condition
    TEMPERATURE = 1.2                   # Set the temperature for sampling

    # Load a specific condition embedding (e.g., from the training set)
    generated_seqs_from_original = []
    for i in range(len(dataset.embeddings)):
        sample_cond = dataset.embeddings[i] 
        for j in range(NUM_GENERATIONS_PER_CONDITION):
            generated_seq = generate_sequence(model, sample_cond, DEVICE, temperature=TEMPERATURE)
            generated_seqs_from_original.append(generated_seq)
        
    print(f"Generated a total of {len(generated_seqs_from_original)} sequences ({len(dataset.embeddings)} conditions * {NUM_GENERATIONS_PER_CONDITION} generations each).")

    # --- Post-generation processing: Deduplication and CSV export ---
    print("\nProcessing generated sequences...")
    print(f"Number of sequences before deduplication: {len(generated_seqs_from_original)}")

    # 1. Remove duplicates while preserving order
    seen = set()
    unique_generated_seqs = []
    for seq in generated_seqs_from_original:
        if seq not in seen:
            seen.add(seq)
            unique_generated_seqs.append(seq)

    print(f"Number of sequences after deduplication: {len(unique_generated_seqs)}")

    # 2. Create DataFrame and save to CSV
    # Prepare data for the DataFrame
    num_unique_seqs = len(unique_generated_seqs)
    seq_ids = list(range(1, num_unique_seqs + 1))  # seq_id from 1 to N
    seq_types = ['gene'] * num_unique_seqs         # All seq_type are 'gene'
    labels = [1] * num_unique_seqs                 # All labels are 1

    df_data = {
        'seq_id': seq_ids,
        'seq_type': seq_types,
        'seq': unique_generated_seqs,
        'label': labels
    }

    df = pd.DataFrame(df_data)

    # Define the output CSV file path
    OUTPUT_CSV_PATH = os.path.join(CHECKPOINT_DIR, f"generated_promoters_{num_unique_seqs}.csv")

    # Save the DataFrame to CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Unique generated sequences saved to CSV: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()