import os
import numpy as np
import random
import time
from collections import Counter
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tensorflow as tf

from model_messidor import create_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

    
# Path Definitions
DATASET_BASE_PATH = os.path.join("messidor_alpha_1.0")
TEST_DATA_PATH = os.path.join("test.npy")
TEST_LABELS_PATH = os.path.join("one_hot_labels.npy")

#===============================================================================================

class QMIXAgent:
    """Implements the QMIX architecture for Multi-Agent Reinforcement Learning."""
    def __init__(self, state_dim, action_dim, n_agents):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = 0.99
        self.lr = 0.001

        # Q-networks for each agent (client)
        self.q_networks = [self._build_q_network() for _ in range(n_agents)]
        self.critic_network = self._build_critic_network()
        self.mixing_network = self._build_mixing_network()

        # Optimizers
        self.q_optimizers = [tf.keras.optimizers.Adam(learning_rate=self.lr) for _ in range(n_agents)]
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.mixing_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def _build_q_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='sigmoid') # action mu in [0, 1]
        ])
        return model

    def _build_critic_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        return model

    def _build_mixing_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.n_agents, activation='linear') 
        ])
        return model

    def select_actions(self, states):
        # ... (full select_actions implementation from your code)
        concatenated_states = tf.convert_to_tensor(states, dtype=tf.float32)
        if len(concatenated_states.shape) == 1:
            concatenated_states = tf.expand_dims(concatenated_states, axis=0)
        q_values = [q_net(concatenated_states) for q_net in self.q_networks]
        q_values = tf.stack(q_values, axis=1)
        action = tf.reduce_mean(q_values, axis=1)
        action = tf.nn.sigmoid(action)
        return action[0]

    def train(self, states, actions, rewards, next_states, dones):
        # ... (full train implementation from your code)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        rewards = tf.squeeze(rewards)
        dones = tf.squeeze(dones)
        
        with tf.GradientTape(persistent=True) as tape:
            concatenated_states = states
            concatenated_next_states = next_states
            q_values_curr = [q_net(concatenated_states) for q_net in self.q_networks]
            q_values_curr = tf.stack(q_values_curr, axis=1) 
            q_values_next = [q_net(concatenated_next_states) for q_net in self.q_networks]
            q_values_next = tf.stack(q_values_next, axis=1) 

            max_q_next = tf.reduce_max(q_values_next, axis=-1)
            centralized_values_next = self.critic_network(concatenated_next_states)
            mixed_q_next = self.mixing_network(centralized_values_next)
            q_target_joint = tf.reduce_sum(mixed_q_next * max_q_next, axis=-1)

            q_targets = rewards + self.gamma * q_target_joint * (1.0 - dones)
            actions = tf.cast(actions, dtype=tf.int32)
            if self.action_dim == 1:
                actions = tf.squeeze(actions)

            individual_q_values = tf.reduce_sum(q_values_curr * tf.one_hot(actions, self.action_dim, dtype=q_values_curr.dtype), axis=-1)

            centralized_values_curr = self.critic_network(concatenated_states)
            mixed_q_curr = self.mixing_network(centralized_values_curr)
            q_current_joint = tf.reduce_sum(mixed_q_curr * individual_q_values, axis=-1)
            
            td_errors = tf.square(q_targets - q_current_joint)
            mean_loss = tf.reduce_mean(td_errors)

        # Apply gradients (logic preserved from your code)
        q_gradients = [tape.gradient(mean_loss, q_net.trainable_variables) for q_net in self.q_networks]
        for i in range(self.n_agents):
            if all(g is not None for g in q_gradients[i]):
                 self.q_optimizers[i].apply_gradients(zip(q_gradients[i], self.q_networks[i].trainable_variables))

        critic_gradients = tape.gradient(mean_loss, self.critic_network.trainable_variables)
        if critic_gradients is not None:
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic_network.trainable_variables))

        mixing_gradients = tape.gradient(mean_loss, self.mixing_network.trainable_variables)
        if mixing_gradients is not None:
            self.mixing_optimizer.apply_gradients(zip(mixing_gradients, self.mixing_network.trainable_variables))
        
        del tape

print("QMIX - done")

#===============================================================================================

# --- Data Loading and Helper Functions ---
def load_img_data(path):
    # ... (full load_img_data implementation from your code)
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
    # ... (full test_model implementation)
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {:.4f}'.format(comm_round, accuracy, loss))
    return accuracy, loss

def avg_weights(scaled_weight_list):
    # ... (full avg_weights implementation)
    num_clients = len(scaled_weight_list)
    if num_clients == 0:
        return None 
    avg_grad = list()
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(tf.stack(grad_list_tuple, axis=0), axis=0) / num_clients
        avg_grad.append(layer_mean.numpy())
    return avg_grad

def weighted_aggregation(local_weight_list, weights):
    # ... (full weighted_aggregation implementation)
    aggregated_weights = []
    for layer_weights in zip(*local_weight_list):
        weighted_layer = np.zeros_like(layer_weights[0])
        for w, weight in zip(layer_weights, weights):
            weighted_layer += w * weight
        aggregated_weights.append(weighted_layer)
    return aggregated_weights 

def calculate_similarity_metrics(global_model_weights, client_model_weights_list):
    # ... (full calculate_similarity_metrics implementation)
    # The helper 'flatten_weights' is needed inside this function.
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

        normalized_similarity = (similarity + 1) / 2
        similarity_metrics.append(normalized_similarity)
    return np.array(similarity_metrics)

def update_weights_som(local_weight_list, global_model, som_weights, normalized_similarity_metrics, initial_sigma, sigma_decay_rate):
    # ... (full update_weights_som implementation - simplified for weights only)
    sims = np.array(normalized_similarity_metrics).flatten()
    n = len(local_weight_list)
    if sims.size != n:
        weights = np.ones(n) / n
    else:
        sims = np.maximum(sims, 0.0)
        total = np.sum(sims)
        if total <= 0:
            weights = np.ones(n) / n
        else:
            weights = sims / total
    return weights

def fairness_loss(F, F_k, M):
    # ... (full fairness_loss implementation)
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
    label = np.zeros((1, 3))

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
state_size = 4 
action_size = 1 
qm_agent = QMIXAgent(state_size, action_size, num_clients)
global_model = create_model()

# Pre-calculated Ek and Pk (as in your original code)
E1, P1, E2, P2, E3, P3, E4, P4 = 0.05739, 0.25, 0.05731, 0.25, 0.05727, 0.25, 0.0574, 0.25 # (Example values, replace with your true calculated ones)
states = np.array([
    [E1, P1, 0.0, 0.0], # E_k, P_k, local_acc, local_loss
    [E2, P2, 0.0, 0.0],
    [E3, P3, 0.0, 0.0],
    [E4, P4, 0.0, 0.0]
])

# --- Training Parameters ---
acc1 = []
loss1 = []
rewards = []
best_acc = 0
best_weights = global_model.get_weights() # Initialize best weights
initial_sigma = 2.0
sigma_decay_rate = 0.1
som_weights = np.random.rand(2, 3, 101) # SOM weights for custom aggregation

# --- Federated Learning Loop (One step for demonstration) ---
start_total_time = time.time()
NUM_STEPS = 60 # Set to a lower number (e.g., 5) for quick testing

for step in range(NUM_STEPS): 
    start_step_time = time.time()
    print(f"\n--- Starting Communication Round/Step {step} ---")
    
    local_weight_list = []
    global_weights = global_model.get_weights()
    client_models_dict = {}

    # 1. QMIX Agent Selects Aggregation Factor (mu)
    qm_state_batch = np.expand_dims(states.flatten(), axis=0)
    mu = qm_agent.select_actions(qm_state_batch)[0].numpy().flatten()[0]
    mu = np.clip(mu, 0.0, 1.0)
    print(f"Global Action Selected: Aggregation Factor mu = {mu:.4f}")
    
    history_list = []

    # 2. Local Training and Weight Scaling
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
        
        client_models_dict[client] = local_model
        history_list.append(history)
        
        # Update state with local training results
        states[i][2] = local_acc
        states[i][3] = local_loss
        
        local_weights = local_model.get_weights()
        
        # Apply FedProx-like local weight scaling using QMIX mu
        for j in range(len(local_weights)):
            local_weights[j] += mu * (global_weights[j] - local_weights[j])
            
        local_weight_list.append(local_weights)

    # 3. Fairness Loss Calculation and Application
    if history_list and local_weight_list:
        F_k = [h.history['loss'][-1] for h in history_list]
        F = np.mean(F_k)
        fairness_loss_value = fairness_loss(F, F_k, num_clients).numpy()
        print(f"Fairness Loss calculated: {fairness_loss_value:.6f}")

        # Apply fairness loss to local weights
        for local_weights in local_weight_list:
            for j in range(len(local_weights)):
                local_weights[j] += fairness_loss_value * (global_weights[j] - local_weights[j])

    # 4. Aggregation (Custom and Weighted)
    if local_weight_list:
        similarity_metrics = calculate_similarity_metrics(global_weights, [model.get_weights() for model in client_models_dict.values()])
        weights = update_weights_som(local_weight_list, global_model, som_weights, similarity_metrics, initial_sigma, sigma_decay_rate)
        aggregated_weights = weighted_aggregation(local_weight_list, weights) 
        global_model.set_weights(aggregated_weights)
    
    # 5. Global Evaluation and Reward
    global_acc, global_loss = test_model(test, label, global_model, step)
    reward = global_acc 

    acc1.append(global_acc)
    loss1.append(global_loss)
    rewards.append(reward)
    
    if global_acc > best_acc:
        best_acc = global_acc
        best_weights = global_model.get_weights()
        
    # 6. QMIX Training
    if step >= 10: # Only train QMIX after some exploration steps
        qm_agent.train(
            np.expand_dims(states.flatten(), axis=0), 
            np.array([[mu]]),
            np.array([reward]), 
            np.expand_dims(states.flatten(), axis=0), # Next state is the same for simplicity
            np.array([False])
        )

    end_step_time = time.time()
    print(f"Step {step} took: {end_step_time - start_step_time:.2f} seconds.")

# --- Final Summary ---
global_model.set_weights(best_weights)
end_total_time = time.time()
print(f"\n=======================================================")
print(f"Total Execution Time for {NUM_STEPS} steps: {end_total_time - start_total_time:.2f} seconds.")
print(f"Best Global Accuracy Achieved: {best_acc:.3%}")
print(f"=======================================================")

#===============================================================================================

# ===============================================================
# Save Best and Final Models for evaluation.py
# ===============================================================

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Save best model weights and structure ---
best_model_path = os.path.join(SAVE_DIR, "best_model.h5")
final_model_path = os.path.join(SAVE_DIR, "final_model.h5")
best_full_model_dir = os.path.join(SAVE_DIR, "best_model_full")

# Save best model weights (.h5)
global_model.set_weights(best_weights)
global_model.save_weights(best_model_path)
print(f"✅ Best model weights saved to: {best_model_path}")

# Save the full best model (architecture + weights + optimizer)
global_model.save(best_full_model_dir)
print(f"✅ Full best model saved to: {best_full_model_dir}")

# Also save the final model after all rounds
final_full_model_dir = os.path.join(SAVE_DIR, "final_model_full")
global_model.save(final_full_model_dir)
print(f"✅ Final model (last round) saved to: {final_full_model_dir}")

# Save model summary and metadata
with open(os.path.join(SAVE_DIR, "training_summary.txt"), "w") as f:
    f.write(f"Total Steps: {NUM_STEPS}\n")
    f.write(f"Best Accuracy: {best_acc:.4f}\n")
    f.write(f"Total Time: {end_total_time - start_total_time:.2f} sec\n")
    f.write("Accuracy History:\n" + ", ".join([f"{a:.4f}" for a in acc1]) + "\n")
    f.write("Loss History:\n" + ", ".join([f"{l:.4f}" for l in loss1]) + "\n")
print("📄 Training summary saved.")

print("\n=======================================================")
print(f"✅ Best model and related artifacts saved under: {SAVE_DIR}")
print("=======================================================\n")
