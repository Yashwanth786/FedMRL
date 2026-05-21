import os
import numpy as np
import random
import time
from collections import Counter
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tensorflow as tf

from model_isic import create_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Input, Lambda
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K


# --- GPU Setup ---
try:
    # Use input() to capture the GPU number
    gpu = int(input("Which gpu number you would like to allocate (0, 1, etc., or -1 for CPU):"))
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"Allocating GPU {gpu}.")
    else:
        # Set all GPU devices to non-visible
        tf.config.set_visible_devices([], 'GPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Forcing TensorFlow to use only the CPU.")
except Exception as e:
    print(f"Could not explicitly configure GPU. Error: {e}. Defaulting to TensorFlow configuration.")


# --- GNN Implementation Utilities ---
def graph_convolution_layer(input_tensor, adj_matrix, output_dim, activation='relu'):
    # Input tensor shape: (Batch_size, N_agents, Input_features)
    # Adj_matrix shape: (Batch_size, N_agents, N_agents)
    
    # 1. Linear Transformation (W * X)
    transformed = Dense(output_dim, use_bias=True)(input_tensor)
    
    # 2. Aggregation (A * WX) - Matrix multiplication with Adjacency Matrix
    def matmul_adj(tensors):
        features = tensors[0]
        adj = tensors[1]
        # Equivalent to: tf.matmul(adj, features)
        return tf.einsum('bij,bjk->bik', adj, features)

    aggregated = Lambda(matmul_adj, output_shape=(None, output_dim))([transformed, adj_matrix])
    
    # 3. Activation
    if activation:
        aggregated = Activation(activation)(aggregated)
        
    return aggregated

# Path Definitions (Assuming these paths are correct in your environment)
DATASET_BASE_PATH = os.path.join("isic_alpha_1.0")
TEST_DATA_PATH = os.path.join("val_images.npy")
TEST_LABELS_PATH = os.path.join("val_labels.npy")

#===============================================================================================

class QMIXAgent:
    """Implements the QMIX architecture with a GNN-Guided Aggregator."""
    def __init__(self, state_dim, action_dim, n_agents):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = 0.99
        self.lr = 0.001
        
        # Q-networks for each agent (client)
        self.q_networks = [self._build_q_network() for _ in range(n_agents)]
        
        # NEW: GNN-Guided Aggregator (replaces the old mixing network logic)
        self.gnn_aggregator = self._build_gnn_aggregator()
        
        # Optimizers
        self.q_optimizers = [tf.keras.optimizers.Adam(learning_rate=self.lr) for _ in range(n_agents)]
        self.gnn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def _build_q_network(self):
        # Input is the individual state vector (state_dim)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear') # Output a single Q-value
        ])
        return model

    def _build_gnn_aggregator(self):
        """
        GNN-based network that takes all client states and the adjacency matrix 
        and outputs the normalized aggregation weights (one per client).
        """
        # Node Features: (N_agents, state_dim)
        input_features = Input(shape=(self.n_agents, self.state_dim), name='Node_Features') 
        # Adjacency Matrix: (N_agents, N_agents)
        input_adj = Input(shape=(self.n_agents, self.n_agents), name='Adj_Matrix') 

        # GNN Layer 1
        x = graph_convolution_layer(input_features, input_adj, 64, activation='relu')
        
        # GNN Layer 2 (Process the contextualized features)
        x = graph_convolution_layer(x, input_adj, 32, activation='relu')

        # Readout Layer: Flatten and combine features
        x_flatten = Flatten()(x)
        
        # Output N_agents aggregation weights
        output_weights = Dense(self.n_agents, 
                               activation='softmax', 
                               name="Aggregation_Weights",
                               # FIX 2: Initialize bias to promote early, even weighting
                               bias_initializer=tf.keras.initializers.Constant(0.1) 
                               )(x_flatten) 
        
        model = Model(inputs=[input_features, input_adj], outputs=output_weights)
        return model

    def get_aggregation_weights(self, states, adj_matrix):
        """Calculates the aggregation weights using the GNN-Aggregator."""
        
        # Ensure correct batch dimensions: (1, N_agents, state_dim) and (1, N_agents, N_agents)
        states = np.expand_dims(states, axis=0)
        adj_matrix = np.expand_dims(adj_matrix, axis=0)

        # The GNN directly outputs the normalized weights
        weights = self.gnn_aggregator([states, adj_matrix], training=False).numpy().flatten()
        return weights

    def train(self, states, adj_matrix, rewards, next_states, next_adj_matrix, dones):
        
        # Ensure inputs are tensors and have batch dimension
        states = tf.convert_to_tensor(np.expand_dims(states, axis=0), dtype=tf.float32)
        adj_matrix = tf.convert_to_tensor(np.expand_dims(adj_matrix, axis=0), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.expand_dims(next_states, axis=0), dtype=tf.float32)
        next_adj_matrix = tf.convert_to_tensor(np.expand_dims(next_adj_matrix, axis=0), dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            
            # --- 1. Calculate Q-values ---
            q_values_curr = [q_net(states[0, i:i+1, :]) for i, q_net in enumerate(self.q_networks)]
            q_values_curr = tf.stack(q_values_curr, axis=1) # Shape: (1, N_agents, 1)

            q_values_next = [q_net(next_states[0, i:i+1, :]) for i, q_net in enumerate(self.q_networks)]
            q_values_next = tf.stack(q_values_next, axis=1) # Shape: (1, N_agents, 1)

            # --- 2. Calculate Aggregation Weights (GNN Output) ---
            A_curr = self.gnn_aggregator([states, adj_matrix]) # Shape: (1, N_agents)
            A_next = self.gnn_aggregator([next_states, next_adj_matrix]) # Shape: (1, N_agents)
            
            # Reshape A to (1, N_agents, 1) for element-wise product
            A_curr_reshaped = tf.expand_dims(A_curr, axis=-1)
            A_next_reshaped = tf.expand_dims(A_next, axis=-1)

            # --- 3. Compute Joint Q-values (Q_total) ---
            q_current_joint = tf.reduce_sum(A_curr_reshaped * q_values_curr, axis=1) # Shape: (1, 1)

            # Target Joint Q-value (Max Q_next is simply the Q_next values since action_dim=1)
            max_q_next = q_values_next 
            q_target_joint = tf.reduce_sum(A_next_reshaped * max_q_next, axis=1) # Shape: (1, 1)
            
            # Calculate Target: R + gamma * Q_target_joint * (1 - done)
            q_targets = rewards + self.gamma * q_target_joint * (1.0 - dones)
            
            # TD Error and Loss
            td_errors = tf.square(q_targets - q_current_joint)
            mean_loss = tf.reduce_mean(td_errors)

        # --- 4. Apply Gradients ---
        
        # Train Q-Networks
        for i in range(self.n_agents):
            q_gradients = tape.gradient(mean_loss, self.q_networks[i].trainable_variables)
            if all(g is not None for g in q_gradients):
                self.q_optimizers[i].apply_gradients(zip(q_gradients, self.q_networks[i].trainable_variables))

        # Train GNN-Aggregator (mixing network)
        gnn_gradients = tape.gradient(mean_loss, self.gnn_aggregator.trainable_variables)
        if all(g is not None for g in gnn_gradients):
            self.gnn_optimizer.apply_gradients(zip(gnn_gradients, self.gnn_aggregator.trainable_variables))
        
        del tape

print("GNN-QMIX Agent - done")

#===============================================================================================

# --- Data Loading and Helper Functions (re-used/simplified) ---

def load_img_data(path):
    img_size = (224, 224)
    datagen = ImageDataGenerator(rescale=1./255.) 
    if not os.path.isdir(path):
        print(f"Error: Client data directory not found at {path}")
        return np.array([]), np.array([])
        
    test_data = datagen.flow_from_directory(
        directory=path,
        target_size=img_size,
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
    )
    total_samples = test_data.n
    test_data.reset()
    images, one_hot_labels = [], []
    i = 0
    for batch in test_data:
        images.extend(batch[0])
        one_hot_labels.extend(batch[1])
        i += 1
        if i >= len(test_data):
            break
    images_np = np.array(images, dtype=np.float32)[:total_samples]
    labels_np = np.array(one_hot_labels, dtype=np.float32)[:total_samples]
    print(f"Loaded {len(images_np)} images from {path}")
    return images_np, labels_np

def test_model(X_test, Y_test, model, comm_round):
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {:.4f}'.format(comm_round, accuracy, loss))
    return accuracy, loss

def weighted_aggregation(local_weight_list, weights):
    """Aggregates weights using GNN-provided weights."""
    aggregated_weights = []
    for layer_weights in zip(*local_weight_list):
        weighted_layer = np.zeros_like(layer_weights[0])
        for w, weight in zip(layer_weights, weights):
            weighted_layer += w * weight
        aggregated_weights.append(weighted_layer)
    return aggregated_weights 

def calculate_similarity_metrics(global_model_weights, client_model_weights_list):
    """Calculates Cosine Similarity (used for Adjacency Matrix)."""
    def flatten_weights(weights_list):
        return np.concatenate([tf.reshape(w, [-1]).numpy() for w in weights_list])

    similarity_metrics = []
    global_weights_flat = flatten_weights(global_model_weights)
    
    for client_model_weights in client_model_weights_list:
        client_weights_flat = flatten_weights(client_model_weights)
        
        global_norm = np.linalg.norm(global_weights_flat)
        client_norm = np.linalg.norm(client_weights_flat)
        if global_norm == 0 or client_norm == 0:
            similarity = 0.0
        else:
            similarity = np.dot(client_weights_flat, global_weights_flat) /(global_norm * client_norm)

        normalized_similarity = (similarity + 1) / 2 # Normalize to [0, 1]
        similarity_metrics.append(normalized_similarity)
    return np.array(similarity_metrics)

def create_adjacency_matrix(client_model_weights_list):
    """
    Creates an Adjacency Matrix where edge weight A[i, j] is the cosine similarity 
    between client i and client j's latest update/model.
    """
    n = len(client_model_weights_list)
    adj_matrix = np.zeros((n, n))
    
    # Flatten weights for all clients
    def flatten_weights(weights_list):
        return np.concatenate([tf.reshape(w, [-1]).numpy() for w in weights_list])
    
    flat_weights = [flatten_weights(w) for w in client_model_weights_list]

    for i in range(n):
        for j in range(i, n):
            # Use dot product normalized by norms
            norm_i = np.linalg.norm(flat_weights[i])
            norm_j = np.linalg.norm(flat_weights[j])
            
            if norm_i == 0 or norm_j == 0:
                similarity = 0.0
            else:
                similarity = np.dot(flat_weights[i], flat_weights[j]) / (norm_i * norm_j)
            
            # Normalize to [0, 1] and set A[i, j] and A[j, i]
            normalized_similarity = (similarity + 1) / 2
            adj_matrix[i, j] = normalized_similarity
            adj_matrix[j, i] = normalized_similarity

    # Add self-loops (diagonal = 1)
    np.fill_diagonal(adj_matrix, 1.0) 
    
    return adj_matrix

def fairness_loss(F, F_k, M):
    F_k = tf.convert_to_tensor(F_k, dtype=tf.float32)
    F_w = tf.reduce_mean(F_k)
    term = tf.reduce_sum(tf.square(F_k - F_w))
    return term / M

#===============================================================================================

# --- Data Loading and Preprocessing ---
print("\n--- Data Loading ---")
train1, label1 = load_img_data(os.path.join(DATASET_BASE_PATH, 'client_1'))
train2, label2 = load_img_data(os.path.join(DATASET_BASE_PATH, 'client_2'))
train3, label3 = load_img_data(os.path.join(DATASET_BASE_PATH, 'client_3'))
train4, label4 = load_img_data(os.path.join(DATASET_BASE_PATH, 'client_4'))

try:
    test = np.load(TEST_DATA_PATH) / 255.0
    label = np.load(TEST_LABELS_PATH)
    print("Test data import successful and normalized.")
except FileNotFoundError:
    print("Error: Test data files not found. Using placeholder data.")
    test = np.zeros((1, 224, 224, 3))
    label = np.zeros((1, 5)) # Must match model output classes

# --- Client Data Maps ---
client_data2 = {
    'client1': (train1, label1),
    'client2': (train2, label2),
    'client3': (train3, label3),
    'client4': (train4, label4),
}

test_batched = {
    'client1': (test, label),
    'client2': (test, label),
    'client3': (test, label),
    'client4': (test, label)
}

# --- QMIX and State Initialization ---
num_clients = 4
state_size = 4  # E_k, P_k, local_acc, local_loss (Node features)
action_size = 1 # Single Q-value output per agent (since action is implicit)
qm_agent = QMIXAgent(state_size, action_size, num_clients)
global_model = create_model()

# Pre-calculated Ek and Pk (Example values)
E1, P1, E2, P2, E3, P3, E4, P4 = 0.05739, 0.25, 0.05731, 0.25, 0.05727, 0.25, 0.0574, 0.25 

# State matrix: (N_agents, state_dim)
states = np.array([
    [E1, P1, 0.0, 0.0], # E_k, P_k, local_acc, local_loss
    [E2, P2, 0.0, 0.0],
    [E3, P3, 0.0, 0.0],
    [E4, P4, 0.0, 0.0]
], dtype=np.float32)

# Initialize Adjacency Matrix (Identity at start)
adj_matrix = np.identity(num_clients, dtype=np.float32)

# --- Training Parameters ---
acc1 = []
loss1 = []
rewards = []
best_acc = 0
best_weights = global_model.get_weights() 
NUM_STEPS = 60 

# Hyperparameters for the combined reward
ALPHA = 1.0  # Weight for Global Accuracy
BETA = 0.5   # Weight for Fairness Loss (Penalty)

# --- FIX 1: Exploration Hyperparameter ---
EXPLORATION_EPSILON = 0.1 
EXPLORATION_DECAY = 0.95
EXPLORATION_STEPS = 10 

# --- Federated Learning Loop ---
start_total_time = time.time()
previous_states = states.copy()
previous_adj_matrix = adj_matrix.copy()

for step in range(NUM_STEPS): 
    start_step_time = time.time()
    print(f"\n--- Starting Communication Round/Step {step} ---")
    
    local_weight_list = []
    local_models_weights = [] # Stores unscaled weights for adj matrix calculation
    global_weights = global_model.get_weights()
    history_list = []
    current_local_losses = []
    
    # 1. Local Training and Weight Collection
    for i, (client, (train_data, label_data)) in enumerate(client_data2.items()):
        local_model = create_model()
        local_model.set_weights(global_weights)

        if train_data.size == 0 or label_data.size == 0:
            print(f"Client {client}: Skipping due to empty data.")
            continue

        history = local_model.fit(train_data, label_data, validation_data=(test_batched[client][0], test_batched[client][1]), epochs=1, batch_size=32, verbose=0)
        
        local_acc = history.history['accuracy'][-1]
        local_loss = history.history['loss'][-1]
        print(f"Client {client}: Acc: {local_acc:.3%} | Loss: {local_loss:.4f}")
        
        history_list.append(history)
        current_local_losses.append(local_loss)
        
        # Update state matrix with new metrics
        states[i][2] = local_acc
        states[i][3] = local_loss
        
        local_weights = local_model.get_weights()
        local_weight_list.append(local_weights)
        local_models_weights.append(local_weights) # Unscaled weights for aggregation

    # 2. Update Adjacency Matrix
    if local_models_weights:
        new_adj_matrix = create_adjacency_matrix(local_models_weights)
        adj_matrix = new_adj_matrix
    else:
        print("No clients updated weights. Skipping aggregation.")
        # Ensure states and adj_matrix remain unchanged for QMIX transition
        previous_states = states.copy()
        previous_adj_matrix = adj_matrix.copy()
        continue

    # 3. GNN-Guided Aggregation
    if local_weight_list:
        # GNN provides the normalized aggregation weights
        gnn_weights = qm_agent.get_aggregation_weights(states, adj_matrix)
        
        # --- FIX 1: Add Exploration Noise to GNN Weights ---
        if step < EXPLORATION_STEPS and np.random.rand() < EXPLORATION_EPSILON:
            noise = np.random.uniform(-0.1, 0.1, size=gnn_weights.shape)
            gnn_weights = gnn_weights + noise
            # Re-normalize to ensure weights sum to 1
            gnn_weights = np.clip(gnn_weights, 0.0, 1.0)
            gnn_weights = gnn_weights / np.sum(gnn_weights)
            print(f"  [EXPLORATION APPLIED] GNN Weights (Noisy): {', '.join([f'{w:.4f}' for w in gnn_weights])}")
        
        print(f"GNN Aggregation Weights (Final): {', '.join([f'{w:.4f}' for w in gnn_weights])}")
        
        aggregated_weights = weighted_aggregation(local_weight_list, gnn_weights) 
        global_model.set_weights(aggregated_weights)
    
    # 4. Global Evaluation and Reward Calculation
    global_acc, global_loss = test_model(test, label, global_model, step)
    
    # Calculate Fairness Loss
    if current_local_losses:
        F_k = current_local_losses
        F = np.mean(F_k)
        fairness_loss_value = fairness_loss(F, F_k, num_clients).numpy()
        print(f"Fairness Loss calculated: {fairness_loss_value:.6f}")
    else:
        fairness_loss_value = 0.0

    # Composite Reward: Maximize Global Accuracy, Minimize Fairness Loss
    reward = (ALPHA * global_acc) - (BETA * fairness_loss_value)
    print(f"QMIX Reward: {reward:.4f}")

    acc1.append(global_acc)
    loss1.append(global_loss)
    rewards.append(reward)
    
    if global_acc > best_acc:
        best_acc = global_acc
        best_weights = global_model.get_weights()
        
    # 5. QMIX Training
    if step >= 1: # Start training QMIX after the first round
        qm_agent.train(
            previous_states, 
            previous_adj_matrix,
            np.array([reward]), 
            states, 
            adj_matrix,
            np.array([False])
        )
        # Decay epsilon slightly to reduce exploration over time
        EXPLORATION_EPSILON *= EXPLORATION_DECAY

    # 6. Update previous states for next round's QMIX transition
    previous_states = states.copy()
    previous_adj_matrix = adj_matrix.copy()

    end_step_time = time.time()
    print(f"Step {step} took: {end_step_time - start_step_time:.2f} seconds.")

# --- Final Summary ---
global_model.set_weights(best_weights)
end_total_time = time.time()
print(f"\n=======================================================")
print(f"Total Execution Time for {NUM_STEPS} steps: {end_total_time - start_total_time:.2f} seconds.")
print(f"Best Global Accuracy Achieved: {best_acc:.3%}")
print(f"=======================================================")

# ===============================================================
# Save Best and Final Models for evaluation.py
# ===============================================================

SAVE_DIR = "saved_models_gnn"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save best model weights
best_model_path = os.path.join(SAVE_DIR, "best_model.h5")
global_model.set_weights(best_weights)
global_model.save_weights(best_model_path)
print(f"✅ Best model weights saved to: {best_model_path}")

# Save the full best model
best_full_model_dir = os.path.join(SAVE_DIR, "best_model_full")
global_model.save(best_full_model_dir)
print(f"✅ Full best model saved to: {best_full_model_dir}")

# Save model summary and metadata
with open(os.path.join(SAVE_DIR, "training_summary.txt"), "w") as f:
    f.write(f"Total Steps: {NUM_STEPS}\n")
    f.write(f"Best Accuracy: {best_acc:.4f}\n")
    f.write(f"Total Time: {end_total_time - start_total_time:.2f} sec\n")
    f.write("Accuracy History:\n" + ", ".join([f"{a:.4f}" for a in acc1]) + "\n")
    f.write("Loss History:\n" + ", ".join([f"{l:.4f}" for l in loss1]) + "\n")
print("📄 Training summary saved.")

print("\n=======================================================")
print(f"✅ GNN-QMIX model and related artifacts saved under: {SAVE_DIR}")
print("=======================================================\n")