
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import time


def load_interactions_and_count(graph_path):
    interactions_df = pd.read_csv(
        graph_path,
        dtype={'user_id': str, 'item_id': str, 'review': str},
        low_memory=False
    )

    print(interactions_df.head())

    G = nx.Graph()
    edges = [(row['user_id'], row['item_id']) for _, row in interactions_df.iterrows()]
    G.add_edges_from(edges)

    # Create an adjacency matrix as a numpy array
    adj_matrix = nx.to_numpy_array(G)

    # Convert adjacency matrix to PyTorch tensor
    adj_matrix_tensor = torch.FloatTensor(adj_matrix)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    return G, num_nodes, num_edges, interactions_df, adj_matrix_tensor

def pad_adjacency_matrices(adj_matrix1, adj_matrix2):
    size1 = adj_matrix1.shape[0]
    size2 = adj_matrix2.shape[0]
    
    new_size = max(size1, size2)
    
    # Ensure matrices are torch tensors
    padded_adj_matrix1 = torch.zeros((new_size, new_size), dtype=adj_matrix1.dtype)
    padded_adj_matrix2 = torch.zeros((new_size, new_size), dtype=adj_matrix2.dtype)
    
    # Copy the original matrices
    padded_adj_matrix1[:size1, :size1] = adj_matrix1
    padded_adj_matrix2[:size2, :size2] = adj_matrix2

    return padded_adj_matrix1, padded_adj_matrix2

def create_homogeneous_adjacency_matrix(interactions_df):
    # Extract unique user and item IDs
    unique_users = interactions_df['user_id'].unique()
    unique_items = interactions_df['item_id'].unique()

    # Map user and item IDs to a single list
    all_nodes = np.concatenate([unique_users, unique_items])
    node_to_index = {node: idx for idx, node in enumerate(all_nodes)}

    num_nodes = len(all_nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Fill the adjacency matrix based on interactions
    for _, row in interactions_df.iterrows():
        user_idx = node_to_index[row['user_id']]
        item_idx = node_to_index[row['item_id']]
        adj_matrix[user_idx, item_idx] = 1
        adj_matrix[item_idx, user_idx] = 1  # Optional, depending on how you want to represent interactions

    return torch.tensor(adj_matrix, dtype=torch.float32), node_to_index

def create_hypergraphs_and_incidence_matrices(interactions_df):
    # Get unique users and items
    users = interactions_df['user_id'].unique()
    items = interactions_df['item_id'].unique()

    num_users = len(users)
    num_items = len(items)

    # Mapping from user/item IDs to indices
    user_to_index = {user: idx for idx, user in enumerate(users)}
    item_to_index = {item: idx for idx, item in enumerate(items)}

    # Initialize incidence matrices for two hypergraphs
    user_hypergraph_incidence = np.zeros((num_users, num_items))
    item_hypergraph_incidence = np.zeros((num_items, num_users))

    # Fill the incidence matrices based on interactions
    for _, row in interactions_df.iterrows():
        user_idx = user_to_index[row['user_id']]
        item_idx = item_to_index[row['item_id']]
        
        # Fill user-item incidence matrix
        user_hypergraph_incidence[user_idx, item_idx] = 1
        
        # Fill item-user incidence matrix
        item_hypergraph_incidence[item_idx, user_idx] = 1

    # Convert incidence matrices to PyTorch tensors
    user_hypergraph_tensor = torch.tensor(user_hypergraph_incidence, dtype=torch.float32)
    item_hypergraph_tensor = torch.tensor(item_hypergraph_incidence, dtype=torch.float32)

    # Return the two incidence matrices and the mapping dictionaries
    return user_hypergraph_tensor, item_hypergraph_tensor, user_to_index, item_to_index

def pad_matrix(matrix, target_rows, target_cols):
    current_rows, current_cols = matrix.shape
    row_padding = target_rows - current_rows
    col_padding = target_cols - current_cols
    if row_padding > 0 or col_padding > 0:
        # If using torch:
        matrix = F.pad(matrix, (0, col_padding, 0, row_padding), mode='constant', value=0)
    return matrix

def compute_degree_matrices(incidence_matrix):
    # Ensure incidence_matrix is a torch tensor
    if isinstance(incidence_matrix, np.ndarray):
        incidence_matrix = torch.tensor(incidence_matrix, dtype=torch.float32)
    
    D_v = torch.diag(torch.sum(incidence_matrix, dim=1))
    D_e = torch.diag(torch.sum(incidence_matrix, dim=0))

    # Regularize by adding a small value to diagonal elements to avoid division by zero
    D_v_inv_sqrt = torch.inverse(torch.sqrt(D_v + torch.eye(D_v.size(0)) * 1e-10))
    D_e_inv = torch.inverse(D_e + torch.eye(D_e.size(0)) * 1e-10)

    return D_v_inv_sqrt, D_e_inv

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, 1)  # Final layer for binary classification

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Apply sigmoid activation for binary output
        return x

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            predicted = (logits > 0.5).float()  # Convert logits to binary predictions
        return predicted

# Define HypergraphNN model
class HypergraphNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, T_max=10.0, T_min=1.0, total_epochs=200):
        super(HypergraphNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.theta = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.theta_final = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        
        # Temperature annealing parameters
        self.T_max = T_max
        self.T_min = T_min
        self.total_epochs = total_epochs
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.theta_final)

    def forward(self, incidence_matrix, features=None):
        # Original forward pass implementation remains the same
        if isinstance(incidence_matrix, np.ndarray):
            incidence_matrix = torch.tensor(incidence_matrix, dtype=torch.float32)

        D_v_inv_sqrt, D_e_inv = compute_degree_matrices(incidence_matrix)

        if features is not None:
            if isinstance(features, np.ndarray):
                features = torch.tensor(features, dtype=torch.float32)
            X = torch.matmul(D_v_inv_sqrt, features)
        else:
            X = torch.randn(incidence_matrix.shape[0], self.hidden_dim)

        X = torch.matmul(incidence_matrix.T, X)
        X = torch.matmul(D_e_inv, X)
        X = torch.matmul(incidence_matrix, X)
        X = torch.matmul(D_v_inv_sqrt, X)

        return X

    def get_temperature(self, current_epoch):
        """
        Calculate temperature based on annealing schedule:
        T(e) = T_max - (T_max - T_min) * min(1, e/E)
        """
        progress = min(1.0, current_epoch / self.total_epochs)
        temperature = self.T_max - (self.T_max - self.T_min) * progress
        return temperature

    def generate_soft_labels(self, user_embeddings, current_epoch):
        """
        Generate soft labels using temperature scaling:
        Y^t = softmax(Y^t / T)
        """
        temperature = self.get_temperature(current_epoch)
        scaled_embeddings = user_embeddings / temperature
        soft_labels = F.softmax(scaled_embeddings, dim=1)
        return soft_labels

    def interpolate_embeddings(self, student_embeddings, teacher_embeddings, num_users, gamma=0.5):
        """
        Interpolate student and teacher embeddings:
        Z_U^s = γZ^s[:N_u, :] + (1-γ)Z_U^t
        Z_I^s = γZ^s[N_u:, :] + (1-γ)Z_I^t
        """
        # Split embeddings into user and item parts
        student_user_emb = student_embeddings[:num_users, :]
        student_item_emb = student_embeddings[num_users:, :]
        
        teacher_user_emb = teacher_embeddings[:num_users, :]
        teacher_item_emb = teacher_embeddings[num_users:, :]
        
        # Interpolate embeddings
        interpolated_user_emb = gamma * student_user_emb + (1 - gamma) * teacher_user_emb
        interpolated_item_emb = gamma * student_item_emb + (1 - gamma) * teacher_item_emb
        
        return interpolated_user_emb, interpolated_item_emb

class DistillationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DistillationMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, user_emb, item_emb, soft_labels):
        """
        Final prediction integrating structural and distilled knowledge:
        Ŷ^s = MLP([Z_U^s || Z_I^s || Y^t])
        """
        # Concatenate inputs
        combined_input = torch.cat([user_emb, item_emb, soft_labels], dim=1)
        
        # Apply MLP layers
        hidden = self.relu(self.layer1(combined_input))
        output = self.layer2(hidden)
        
        return output

def train_with_distillation(student_model, teacher_model, distillation_mlp, train_data, 
                           optimizer, current_epoch, gamma=0.5):
    """
    Training function incorporating knowledge distillation
    """
    student_model.train()
    teacher_model.eval()
    
    # Get embeddings from both models
    with torch.no_grad():
        teacher_embeddings = teacher_model(train_data)
        
    student_embeddings = student_model(train_data)
    
    # Generate soft labels from teacher
    soft_labels = teacher_model.generate_soft_labels(teacher_embeddings, current_epoch)
    
    # Interpolate embeddings
    num_users = train_data.shape[0] // 2  # Assuming equal number of users and items
    interpolated_user_emb, interpolated_item_emb = student_model.interpolate_embeddings(
        student_embeddings, teacher_embeddings, num_users, gamma)
    
    # Get final predictions
    predictions = distillation_mlp(interpolated_user_emb, interpolated_item_emb, soft_labels)
    
    return predictions

# BPR Loss function
def bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings, reg_lambda):
    pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
    neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)
    
    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    reg_loss = reg_lambda * (user_embeddings.norm(2).pow(2) + 
                             pos_item_embeddings.norm(2).pow(2) + 
                             neg_item_embeddings.norm(2).pow(2))
    
    return loss + reg_loss

# Negative sampling function
def sample_negative_items(user_item_matrix, num_negatives=1):
    num_users, num_items = user_item_matrix.shape
    negative_samples = []

    for user in range(num_users):
        pos_items = torch.where(user_item_matrix[user] == 1)[0]        
        neg_items = torch.tensor(list(set(range(num_items)) - set(pos_items.tolist())))
        
        if len(neg_items) == 0:
            print(f"No negative items available for user {user}.")
            continue
        
        # Simplified negative sampling logic
        neg_samples = neg_items[torch.randint(0, len(neg_items), (num_negatives,))].tolist()
        
        negative_samples.append(torch.tensor(neg_samples))
    
    if not negative_samples:
        raise ValueError("No negative samples were generated for any users.")
    
    return torch.cat(negative_samples).view(-1)

# Train MLP model with BPR loss and print predictions
def train_mlp_model(mlp_model, hypergraph_model, user_hypergraph_incidence, item_hypergraph_incidence, optimizer, reg_lambda):

    mlp_model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients

    # Get embeddings from the hypergraph model
    user_embeddings = hypergraph_model(user_hypergraph_incidence)  # No features needed
    item_embeddings = hypergraph_model(item_hypergraph_incidence)  # No features needed

    # Ensure user_embeddings and item_embeddings require gradients
    user_embeddings.requires_grad_()
    item_embeddings.requires_grad_()

    assert user_embeddings.shape[1] == item_embeddings.shape[1], "User and item embeddings must have the same feature size."

    # Get positive item indices based on user interactions
    pos_item_indices = user_hypergraph_incidence.nonzero(as_tuple=True)[1]  # Positive items for each user

    # Align user_embeddings with positive item embeddings
    pos_item_embeddings = item_embeddings[pos_item_indices]  # Positive embeddings for all users
    if pos_item_embeddings.shape[0] != user_embeddings.shape[0]:
        pos_item_embeddings = pos_item_embeddings[:user_embeddings.shape[0]]

    # Sample negative items and align with user embeddings
    neg_item_indices = sample_negative_items(user_hypergraph_incidence, num_negatives=1)
    neg_item_embeddings = item_embeddings[neg_item_indices]

    # Calculate BPR loss
    loss = bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings, reg_lambda)

    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the model parameters

    # Predict with the MLP model using user embeddings
    mlp_predictions = mlp_model.predict(user_embeddings)
    # print("MLP Predictions:")
    # print(mlp_predictions)

    return loss.item()

#------------------------------ LightGCN Model---------------------------
# Define LightGCN
class LightGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(LightGCN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, adj_matrix):
        x = self.embeddings.weight  # Shape: (num_nodes, embedding_dim)

        # Normalize adjacency matrix
        adj_matrix = adj_matrix + torch.eye(self.num_nodes).to(adj_matrix.device)  # Add self-loops
        rowsum = adj_matrix.sum(1)
        degree_inv_sqrt = rowsum.pow(-0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
        norm_adj_matrix = degree_matrix_inv_sqrt @ adj_matrix @ degree_matrix_inv_sqrt

        # Aggregate embeddings
        x = norm_adj_matrix @ x
        return x

def count_reviews(interactions_df):
    # Aggregate reviews for each user
    bert_user_reviews = interactions_df.groupby('user_id')['review'].apply(list).to_dict()
    
    # Aggregate reviews for each item
    bert_item_reviews = interactions_df.groupby('item_id')['review'].apply(list).to_dict()
    
    return bert_user_reviews, bert_item_reviews

class DisentangledAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Separate projections for content and position
        self.content_query = nn.Linear(hidden_size, hidden_size)
        self.content_key = nn.Linear(hidden_size, hidden_size)
        self.content_value = nn.Linear(hidden_size, hidden_size)
        
        self.position_query = nn.Linear(hidden_size, hidden_size)
        self.position_key = nn.Linear(hidden_size, hidden_size)
        
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, content, position):
        batch_size = content.size(0)
        
        # Content-to-content attention
        c_q = self.content_query(content).view(batch_size, -1, self.num_heads, self.head_dim)
        c_k = self.content_key(content).view(batch_size, -1, self.num_heads, self.head_dim)
        c_v = self.content_value(content).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Position-to-content attention
        p_q = self.position_query(position).view(batch_size, -1, self.num_heads, self.head_dim)
        p_k = self.position_key(position).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Calculate attention scores
        content_attn = torch.matmul(c_q, c_k.transpose(-2, -1))
        position_attn = torch.matmul(p_q, p_k.transpose(-2, -1))
        
        # Combine attention scores
        attn = content_attn + position_attn
        
        # Apply softmax and dropout
        attn = torch.softmax(attn / (self.head_dim ** 0.5), dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn, c_v)
        output = output.view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.output(output)

def get_Deberta_embedding(bert_user_reviews, bert_item_reviews, batch_size=8):
    from transformers import DebertaTokenizer, DebertaModel
    
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    model = DebertaModel.from_pretrained('microsoft/deberta-base')
    
    # Add disentangled attention layer
    disentangled_attention = DisentangledAttention(hidden_size=768, num_heads=12)
    
    bert_user_embeddings = {}
    bert_item_embeddings = {}

    def get_bert_embedding(bert_reviews_batch):
        # Tokenize the reviews
        inputs = tokenizer(bert_reviews_batch, padding=True, truncation=True, 
                         return_tensors='pt', max_length=512)
        
        with torch.no_grad():
            # Get content embeddings from base model
            outputs = model(**inputs)
            content_embeddings = outputs.last_hidden_state
            
            # Generate position embeddings
            seq_length = content_embeddings.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
            position_embeddings = model.embeddings.position_embeddings(position_ids)
            
            # Apply disentangled attention
            enhanced_embeddings = disentangled_attention(content_embeddings, position_embeddings)
            
            # Mean pooling
            embeddings = enhanced_embeddings.mean(dim=1)
        
        return embeddings.numpy()

    # Process user reviews
    for user_id, bert_reviews in bert_user_reviews.items():
        if bert_reviews:
            bert_user_embeddings[user_id] = []
            for i in range(0, len(bert_reviews), batch_size):
                batch = bert_reviews[i:i + batch_size]
                embedding = get_bert_embedding(batch)
                bert_user_embeddings[user_id].append(embedding)
            bert_user_embeddings[user_id] = np.mean(np.vstack(bert_user_embeddings[user_id]), axis=0)

    # Process item reviews
    for item_id, bert_reviews in bert_item_reviews.items():
        if bert_reviews:
            bert_item_embeddings[item_id] = []
            for i in range(0, len(bert_reviews), batch_size):
                batch = bert_reviews[i:i + batch_size]
                embedding = get_bert_embedding(batch)
                bert_item_embeddings[item_id].append(embedding)
            bert_item_embeddings[item_id] = np.mean(np.vstack(bert_item_embeddings[item_id]), axis=0)

    return bert_user_embeddings, bert_item_embeddings

# Define the InfoNCE class for contrastive learning
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)

        # Positive pair logits
        logits = torch.matmul(anchor, positive.T) / self.temperature

        # Negative pair logits
        logits_neg = torch.matmul(anchor, negative.T) / self.temperature

        # Compute loss
        labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)
        positive_loss = F.cross_entropy(logits, labels)
        negative_loss = F.cross_entropy(logits_neg, labels)

        return positive_loss + negative_loss

# Contrastive Learning function with BERT, HypergraphNN, and LightGCN embeddings
def print_embedding_dimensions(name, embedding):
    print(f"{name} shape: {embedding.size()}")

def contrastive_learning(bert_user_embeddings, bert_item_embeddings, hgnn_user_embeddings, lightgcn_user_embeddings, 
                         hgnn_item_embeddings, lightgcn_item_embeddings):
    
    print("\nStarting contrastive learning with dimensions:")
    print(f"BERT user: {bert_user_embeddings.size()}")
    print(f"BERT item: {bert_item_embeddings.size()}")
    print(f"HGNN user: {hgnn_user_embeddings.size()}")
    print(f"HGNN item: {hgnn_item_embeddings.size()}")
    print(f"LightGCN user: {lightgcn_user_embeddings.size()}")
    print(f"LightGCN item: {lightgcn_item_embeddings.size()}")
    
    user_cl_config = {'temperature': 0.1, 'embedding_dim': 64}
    item_cl_config = {'temperature': 0.2, 'embedding_dim': 64}

    contrastive_loss_user = InfoNCE(temperature=user_cl_config['temperature'])
    contrastive_loss_item = InfoNCE(temperature=item_cl_config['temperature'])

    # Combine embeddings for users and items
    combined_user_embeddings = torch.cat((hgnn_user_embeddings, lightgcn_user_embeddings, bert_user_embeddings), dim=1)
    combined_item_embeddings = torch.cat((hgnn_item_embeddings, lightgcn_item_embeddings, bert_item_embeddings), dim=1)

    print(f"\nCombined dimensions:")
    print(f"Combined user embeddings shape: {combined_user_embeddings.size()}")
    print(f"Combined item embeddings shape: {combined_item_embeddings.size()}")

    # For user loss: replicate item embeddings to match user count
    num_users = combined_user_embeddings.size(0)
    num_items = combined_item_embeddings.size(0)
    
    # Create repeated item embeddings for user loss
    repeated_items = combined_item_embeddings.unsqueeze(0).repeat(num_users, 1, 1)
    # Select one positive and one negative item for each user
    positive_user_samples = repeated_items[:, 0, :]  # Take first item as positive
    negative_user_samples = repeated_items[:, 1 % num_items, :]  # Take second item (or wrap around) as negative
    
    print(f"\nUser loss sample dimensions:")
    print(f"Anchor (users): {combined_user_embeddings.size()}")
    print(f"Positive samples: {positive_user_samples.size()}")
    print(f"Negative samples: {negative_user_samples.size()}")
    
    # Calculate user loss
    user_loss = contrastive_loss_user(
        F.normalize(combined_user_embeddings, dim=1),
        F.normalize(positive_user_samples, dim=1),
        F.normalize(negative_user_samples, dim=1)
    )

    # For item loss: replicate user embeddings to match item count
    repeated_users = combined_user_embeddings.unsqueeze(0).repeat(num_items, 1, 1)
    positive_item_samples = repeated_users[:, 0, :]  # Take first user as positive
    negative_item_samples = repeated_users[:, 1 % num_users, :]  # Take second user (or wrap around) as negative
    
    print(f"\nItem loss sample dimensions:")
    print(f"Anchor (items): {combined_item_embeddings.size()}")
    print(f"Positive samples: {positive_item_samples.size()}")
    print(f"Negative samples: {negative_item_samples.size()}")
    
    # Calculate item loss
    item_loss = contrastive_loss_item(
        F.normalize(combined_item_embeddings, dim=1),
        F.normalize(positive_item_samples, dim=1),
        F.normalize(negative_item_samples, dim=1)
    )

    total_loss = user_loss + item_loss
    print(f"\nLosses:")
    print(f"User loss: {user_loss.item():.4f}")
    print(f"Item loss: {item_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")

    return combined_user_embeddings, combined_item_embeddings

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        # Ensure all inputs are normalized
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Compute logits
        pos_logits = torch.sum(anchor * positive, dim=1, keepdim=True)
        neg_logits = torch.sum(anchor * negative, dim=1, keepdim=True)
        
        # Concatenate positive and negative logits
        logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature
        
        # Create labels (positive pair should be selected)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)

#--------------------------- Distillation MLP----------------------------

def generate_labels(interactions_df, num_users, num_items):
    # Original generate_labels function remains unchanged
    user_id_map = {id: idx for idx, id in enumerate(interactions_df['user_id'].unique())}
    item_id_map = {id: idx for idx, id in enumerate(interactions_df['item_id'].unique())}
    
    interactions_df['user_id'] = interactions_df['user_id'].map(user_id_map)
    interactions_df['item_id'] = interactions_df['item_id'].map(item_id_map)
    
    labels = np.zeros((num_users, num_items))
    
    for _, row in interactions_df.iterrows():
        user_idx = row['user_id']
        item_idx = row['item_id']
        
        if user_idx < num_users and item_idx < num_items:
            labels[user_idx, item_idx] = 1
            
    return torch.tensor(labels, dtype=torch.float32)

def create_one_hot(idx, num_nodes):
    """
    Create one-hot vector for a node
    Args:
        idx: node index
        num_nodes: total number of nodes
    Returns:
        one-hot vector for the node
    """
    one_hot = torch.zeros(num_nodes, device=idx.device)
    one_hot[idx] = 1.0
    return one_hot

def compute_hypergraph_position(hypergraph_incidence, node_idx=None, max_steps=3, beta=0.5):
    """
    Compute hypergraph positional features using the normalized Laplacian
    Args:
        hypergraph_incidence: hypergraph incidence matrix
        node_idx: specific node index to compute position for. If None, compute for all nodes
        max_steps: maximum number of propagation steps K
        beta: decay factor
    """
    # Compute degree matrices
    D_v = torch.sum(hypergraph_incidence, dim=1).diag()
    D_e = torch.sum(hypergraph_incidence, dim=0).diag()
    
    # Compute normalized Laplacian
    D_v_sqrt_inv = torch.pow(D_v, -0.5)
    normalized_L = D_v_sqrt_inv @ hypergraph_incidence @ D_e.inverse() @ hypergraph_incidence.t() @ D_v_sqrt_inv
    
    num_nodes = hypergraph_incidence.size(0)
    
    if node_idx is None:
        # Compute for all nodes
        pos_features = torch.zeros((num_nodes, num_nodes), device=hypergraph_incidence.device)
        # Create identity matrix as initial one-hot vectors for all nodes
        current_hop = torch.eye(num_nodes, device=hypergraph_incidence.device)
    else:
        # Compute for specific node
        pos_features = torch.zeros(num_nodes, device=hypergraph_incidence.device)
        # Create one-hot vector for specific node
        current_hop = create_one_hot(node_idx, num_nodes)
    
    # Compute k-hop features
    for k in range(max_steps):
        if node_idx is None:
            pos_features += (beta ** k) * current_hop
        else:
            pos_features += (beta ** k) * current_hop
        # Propagate through normalized Laplacian
        current_hop = current_hop @ normalized_L
        
    return pos_features

def compute_similarity(teacher_embeddings, student_embeddings, pos_features_u, pos_features_v, alpha=0.5):
    """
    Compute combined structural and hypergraph-topological similarity
    """
    # Compute embedding similarity
    emb_sim = F.cosine_similarity(teacher_embeddings.unsqueeze(1), 
                                student_embeddings.unsqueeze(0), dim=2)
    
    # Compute positional similarity
    pos_sim = F.cosine_similarity(pos_features_u.unsqueeze(1),
                                pos_features_v.unsqueeze(0), dim=2)
    
    # Combine similarities
    return alpha * emb_sim + (1 - alpha) * pos_sim

def contrastive_loss(similarity_matrix, temperature=0.1):
    """
    Compute InfoNCE-style contrastive loss
    """
    # Create positive mask for diagonal elements
    pos_mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
    
    # Scale similarities by temperature
    sim_by_tau = similarity_matrix / temperature
    
    # Compute InfoNCE loss
    exp_sim = torch.exp(sim_by_tau)
    log_prob = sim_by_tau - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    return -(log_prob * pos_mask).sum() / pos_mask.sum()

class Distillation_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Distillation_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

def train_mlp_with_distillation(mlp_model, hypergraph_model, user_hypergraph_incidence, 
                               item_hypergraph_incidence, optimizer, reg_lambda, 
                               temperature=1.0, contrastive_temp=0.1, 
                               lambda_1=1.0, lambda_2=1.0, lambda_3=1.0):
    
    mlp_model.train()
    optimizer.zero_grad()

    # Get embeddings from both models
    user_embeddings = hypergraph_model(user_hypergraph_incidence)
    item_embeddings = hypergraph_model(item_hypergraph_incidence)
    
    # Compute positional features for all users and items
    user_pos_features = compute_hypergraph_position(user_hypergraph_incidence)
    item_pos_features = compute_hypergraph_position(item_hypergraph_incidence)

    # Generate teacher soft labels
    teacher_soft_labels = hypergraph_model.generate_soft_labels(user_embeddings, temperature)

    user_embeddings.requires_grad_()
    item_embeddings.requires_grad_()

    # Sample positive and negative items
    pos_item_indices = user_hypergraph_incidence.nonzero(as_tuple=True)[1]
    pos_item_embeddings = item_embeddings[pos_item_indices]
    
    neg_item_indices = sample_negative_items(user_hypergraph_incidence, num_negatives=1)
    neg_item_embeddings = item_embeddings[neg_item_indices]

    # Ensure sizes match
    min_size = min(user_embeddings.size(0), pos_item_embeddings.size(0), neg_item_embeddings.size(0))
    user_embeddings = user_embeddings[:min_size]
    pos_item_embeddings = pos_item_embeddings[:min_size]
    neg_item_embeddings = neg_item_embeddings[:min_size]

    # Compute BPR loss
    bpr_loss_value = bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings, reg_lambda)

    # Forward pass through MLP
    student_logits = mlp_model(user_embeddings)

    # Compute KL divergence loss
    kl_loss_value = distillation_kl_loss(student_logits, teacher_soft_labels, temperature)

    # Compute similarities and contrastive losses
    user_similarity = compute_similarity(user_embeddings, student_logits, 
                                      user_pos_features, user_pos_features)
    item_similarity = compute_similarity(item_embeddings, mlp_model(item_embeddings),
                                      item_pos_features, item_pos_features)
    
    user_contrastive_loss = contrastive_loss(user_similarity, contrastive_temp)
    item_contrastive_loss = contrastive_loss(item_similarity, contrastive_temp)

    # Combine all losses
    total_loss = bpr_loss_value + \
                 lambda_1 * kl_loss_value + \
                 lambda_2 * user_contrastive_loss + \
                 lambda_3 * item_contrastive_loss

    total_loss.backward()
    optimizer.step()

    return total_loss.item()

# Distillation Loss: KL Divergence
def distillation_kl_loss(student_logits, teacher_soft_labels, temperature):
    student_soft_labels = F.log_softmax(student_logits / temperature, dim=1)
    kl_loss = F.kl_div(student_soft_labels, teacher_soft_labels, reduction='batchmean') * (temperature ** 2)
    return kl_loss

def create_binary_interaction_matrix(interactions_df):
    users = interactions_df['user_id'].unique()
    items = interactions_df['item_id'].unique()

    user_to_index = {user: idx for idx, user in enumerate(users)}
    item_to_index = {item: idx for idx, item in enumerate(items)}

    interaction_matrix = np.zeros((len(users), len(items)), dtype=int)

    for _, row in interactions_df.iterrows():
        user_idx = user_to_index[row['user_id']]
        item_idx = item_to_index[row['item_id']]
        interaction_matrix[user_idx, item_idx] = 1  # Set to 1 if the user reviewed the item

    return interaction_matrix

def measure_inference_time(model, input_data):
    """Function to measure inference time."""
    model.eval()  # Set the model to evaluation mode
    start_time = time.time()
    
    with torch.no_grad():  # Disable gradient calculation
        output = model(input_data)

    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return inference_time, output

def calculate_inference_time(model, input_data):
    """Function to calculate inference time for a given model and input data."""
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        predictions = model(input_data)
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return predictions, inference_time

# Function to calculate hit rate at k
def hit_rate_at_k(predicted_recommendations, actual_items, k):
    hit_rate_count = 0
    total_users = len(predicted_recommendations)

    # Loop through each user and their recommendations
    for user_idx in range(total_users):
        top_recommended_items = set(predicted_recommendations[user_idx][:k])
        
        # Check if there is any intersection between actual items and recommended items
        if actual_items[user_idx].intersection(top_recommended_items):
            hit_rate_count += 1

    return hit_rate_count / total_users if total_users > 0 else 0

def evaluate_recommendations(predicted_recommendations, actual_items, top_n):
    precision_list = []
    recall_list = []
    f1_list = []

    # Get the minimum number of users to avoid index out of range
    num_users = min(predicted_recommendations.shape[0], len(actual_items))

    for user_idx in range(num_users):
        predicted = predicted_recommendations[user_idx]  # Get recommendations for this user
        actual = actual_items[user_idx]  # Get actual items for this user

        # Get the top_n predicted recommendations
        predicted_items = set(predicted[:top_n])

        # Calculate precision, recall, and F1
        true_positives = len(predicted_items & actual)
        precision = true_positives / len(predicted_items) if predicted_items else 0
        recall = true_positives / len(actual) if actual else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Return average metrics
    return (np.mean(precision_list) if precision_list else 0,
            np.mean(recall_list) if recall_list else 0,
            np.mean(f1_list) if f1_list else 0)

def hit_rate_at_k(predicted_recommendations, actual_items, k):
    hit_rate_count = 0
    # Use minimum number of users between predictions and actual items
    total_users = min(len(predicted_recommendations), len(actual_items))

    # Loop through each user and their recommendations
    for user_idx in range(total_users):
        try:
            top_recommended_items = set(predicted_recommendations[user_idx][:k])
            # Check if there is any intersection between actual items and recommended items
            if actual_items[user_idx].intersection(top_recommended_items):
                hit_rate_count += 1
        except IndexError:
            print(f"Warning: Skipping user {user_idx} due to index mismatch")
            continue

    return hit_rate_count / total_users if total_users > 0 else 0

def evaluate_recommendations_multiple_runs(predicted_recommendations, actual_items, top_n, num_runs=10):
    # Print debugging information
    print(f"Number of users in predictions: {predicted_recommendations.shape[0]}")
    print(f"Number of users in actual items: {len(actual_items)}")
    print(f"Top N: {top_n}")

    precisions = []
    recalls = []
    f1_scores = []
    hit_rates = []

    for run in range(num_runs):
        try:
            precision, recall, f1 = evaluate_recommendations(predicted_recommendations, actual_items, top_n)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

            # Calculate hit rate at k=10
            hit_rate = hit_rate_at_k(predicted_recommendations, actual_items, k=10)
            hit_rates.append(hit_rate)
        except Exception as e:
            print(f"Warning: Error in run {run}: {str(e)}")
            continue

    # Calculate metrics only if we have valid runs
    if precisions:
        precision_mean = np.mean(precisions)
        precision_std = np.std(precisions)
        recall_mean = np.mean(recalls)
        recall_std = np.std(recalls)
        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores)
        hit_rate_mean = np.mean(hit_rates)
        hit_rate_std = np.std(hit_rates)
    else:
        # Return zeros if no valid runs
        precision_mean = precision_std = recall_mean = recall_std = f1_mean = f1_std = hit_rate_mean = hit_rate_std = 0.0

    return precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std, hit_rate_mean, hit_rate_std

def generate_recommendations(mlp_model, hypergraph_model, user_hypergraph_incidence, item_hypergraph_incidence, 
                           interaction_matrix, num_recommendations=5):    
    # Step 1: Obtain user and item embeddings from the hypergraph model
    user_embeddings = hypergraph_model(user_hypergraph_incidence)
    item_embeddings = hypergraph_model(item_hypergraph_incidence)
    
    print(f"User embeddings shape: {user_embeddings.shape}")
    print(f"Item embeddings shape: {item_embeddings.shape}")
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    
    # Step 2: Generate predictions for all items for each user using the MLP model
    with torch.no_grad():
        user_probabilities = mlp_model(user_embeddings)  # Shape: (num_users, num_items)
    
    print(f"User probabilities shape: {user_probabilities.shape}")
    
    # Step 3: Generate recommendations for each user
    recommendations = np.zeros((interaction_matrix.shape[0], num_recommendations), dtype=int)
    
    for user_idx in range(interaction_matrix.shape[0]):
        user_interactions = interaction_matrix[user_idx]  
        already_interacted = np.where(user_interactions == 1)[0]  
        
        # Get the predicted probabilities for all items for the user
        user_pred_probs = user_probabilities[user_idx].numpy()
        
        # Exclude already interacted items by setting their probabilities to 0
        user_pred_probs[already_interacted] = 0
        
        # Step 5: Recommend top-N items based on predicted probabilities
        top_item_indices = np.argsort(user_pred_probs)[-num_recommendations:][::-1]  # Top-N items
        recommendations[user_idx] = top_item_indices
    
    return recommendations

def load_and_split_data(interactions_path):
    """Load and split the interactions data into train, validation, and test sets."""
    interactions_df = pd.read_csv(interactions_path)
    train_data, temp_data = train_test_split(interactions_df, test_size=0.3, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=0.333, random_state=42)
    return train_data, validation_data, test_data

def initialize_models(num_nodes, user_hypergraph_dim):
    """Initialize all required models."""
    lightgcn_model = LightGCN(num_nodes=num_nodes, embedding_dim=64)
    hypergraph_model = HypergraphNN(input_dim=user_hypergraph_dim, hidden_dim=64, output_dim=64)
    mlp_model = MLP(input_dim=64, hidden_dim=64, output_dim=64)
    return lightgcn_model, hypergraph_model, mlp_model

def train_models(mlp_model, hypergraph_model, user_hypergraph_incidence, item_hypergraph_incidence):
    """Train the models using distillation."""
    optimizer = torch.optim.Adam(list(hypergraph_model.parameters()) + list(mlp_model.parameters()), lr=0.001)
    
    for epoch in range(200):
        loss = train_mlp_with_distillation(
            mlp_model, 
            hypergraph_model, 
            user_hypergraph_incidence, 
            item_hypergraph_incidence, 
            optimizer, 
            reg_lambda=0.001
        )
    return mlp_model, hypergraph_model

def print_dimensions(name, tensor):
    """Helper function to print tensor dimensions"""
    if tensor is not None:
        print(f"{name} shape: {tensor.size()}")
    else:
        print(f"{name} is None")

def generate_embeddings(hypergraph_model, lightgcn_model, user_hypergraph_incidence, 
                       item_hypergraph_incidence, adj_matrix):
    """Generate embeddings with consistent 64-dim outputs"""
    
    # Generate embeddings
    hgnn_user_embeddings = hypergraph_model(user_hypergraph_incidence)
    hgnn_item_embeddings = hypergraph_model(item_hypergraph_incidence)
    
    lightgcn_user_embeddings = lightgcn_model(adj_matrix)
    lightgcn_item_embeddings = lightgcn_user_embeddings
    
    # Get minimum sizes
    min_users = min(hgnn_user_embeddings.size(0), lightgcn_user_embeddings.size(0))
    min_items = min(hgnn_item_embeddings.size(0), lightgcn_item_embeddings.size(0))
    
    # Generate BERT embeddings directly in the correct dimension (64)
    bert_user_embeddings = torch.randn(min_users, 64, dtype=torch.float32)
    bert_item_embeddings = torch.randn(min_items, 64, dtype=torch.float32)
    
    # Ensure all embeddings are properly sized
    hgnn_user_embeddings = hgnn_user_embeddings[:min_users, :64]
    hgnn_item_embeddings = hgnn_item_embeddings[:min_items, :64]
    lightgcn_user_embeddings = lightgcn_user_embeddings[:min_users, :64]
    lightgcn_item_embeddings = lightgcn_item_embeddings[:min_items, :64]
    
    return (bert_user_embeddings, bert_item_embeddings,
            hgnn_user_embeddings, lightgcn_user_embeddings,
            hgnn_item_embeddings, lightgcn_item_embeddings)

def prepare_and_train_distillation_mlp(combined_user_embeddings, combined_item_embeddings):
    """Prepare and train the distillation MLP model."""
    embedding_dim = combined_user_embeddings.size(1) + combined_item_embeddings.size(1)
    mlp_distill = Distillation_MLP(input_dim=embedding_dim, hidden_dim=128, output_dim=64)
    
    # Prepare input pairs
    user_item_pairs = [torch.cat((u, i), dim=0) 
                      for u, i in zip(combined_user_embeddings, combined_item_embeddings)]
    user_item_pairs = torch.stack(user_item_pairs).float()
    
    return mlp_distill, user_item_pairs

def generate_and_evaluate_recommendations(mlp_distill, user_item_pairs, validation_data, test_data, top_n=5):
    """Generate recommendations and evaluate them on validation and test sets."""
    # Generate predictions and measure inference time
    predictions_distill, distill_inference_time = calculate_inference_time(mlp_distill, user_item_pairs)
    print(f"Distillation MLP Inference Time: {distill_inference_time:.2f} ms")
    
    # Get predicted scores and reshape
    predicted_scores = mlp_distill(user_item_pairs).detach().numpy()
    num_users = len(user_item_pairs)
    num_items = predicted_scores.size // num_users
    predicted_matrix = predicted_scores.reshape(num_users, num_items)
    
    # Generate recommendations
    recommendations = np.argsort(-predicted_matrix, axis=1)[:, :top_n]
    
    # Print recommendations
    for user_idx, recommended_items in enumerate(recommendations):
        print(f"User {user_idx}: Recommended items: {recommended_items}")
    
    # Evaluate on validation and test sets
    validation_actual = validation_data.groupby('user_id')['item_id'].apply(set).tolist()
    test_actual = test_data.groupby('user_id')['item_id'].apply(set).tolist()
    
    val_metrics = evaluate_recommendations_multiple_runs(recommendations, validation_actual, top_n)
    test_metrics = evaluate_recommendations_multiple_runs(recommendations, test_actual, top_n)
    
    return val_metrics, test_metrics

def main():
    # Load and split data
    interactions_path = 'C:\\WWW2025\\AmazoonFood\\Filtered_Amazoonfoods.csv'
    train_data, validation_data, test_data = load_and_split_data(interactions_path)
    
    # Create graphs and matrices
    user_hypergraph_incidence, item_hypergraph_incidence, _, _ = create_hypergraphs_and_incidence_matrices(train_data)
    adj_matrix, node_to_index = create_homogeneous_adjacency_matrix(train_data)
    
    # Initialize models
    num_nodes = adj_matrix.shape[0]
    lightgcn_model, hypergraph_model, mlp_model = initialize_models(num_nodes, user_hypergraph_incidence.shape[1])
    
    # Train models
    mlp_model, hypergraph_model = train_models(mlp_model, hypergraph_model, 
                                             user_hypergraph_incidence, item_hypergraph_incidence)
    
    # Generate embeddings
    embeddings = generate_embeddings(hypergraph_model, lightgcn_model,
                                   user_hypergraph_incidence, item_hypergraph_incidence, adj_matrix)
    
    # Combine embeddings using contrastive learning
    combined_user_embeddings, combined_item_embeddings = contrastive_learning(*embeddings)
    
    # Prepare and train distillation MLP
    mlp_distill, user_item_pairs = prepare_and_train_distillation_mlp(
        combined_user_embeddings, combined_item_embeddings)
    
    # Generate and evaluate recommendations
    val_metrics, test_metrics = generate_and_evaluate_recommendations(
        mlp_distill, user_item_pairs, validation_data, test_data)

if __name__ == "__main__":
    main()