# Setup
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

# Paths
root = Path(__file__).resolve().parent
vad_weight = root / "model" / "best_vad_model_state_dict.pth"
default_i3d_feature = root / "input" / "01_001_i3d.npy"


# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"


# --- VAD model load ---
# Recreate the VAD architecture before loading the trained state_dict
class py_conv1d_block(nn.Module):

    # The input feature is 2048 dimension, initialize with 512 hidden dimension and dropout at 0.5
    # Second conv layer is true at default, but can be set to false when hyperparameter tuning
    def __init__(self, input_dim=2048, hidden_dim=512, dropout=0.5, use_second_conv=True):
        super().__init__()

        self.use_second_conv = use_second_conv

        # First Conv1D layer
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )

        # Second Conv1D layer
        if self.use_second_conv:
            self.conv2 = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
            )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # Define the forward pass of the convolutional block and second conv layer
    def forward(self, x):

        # Change x shape from (batch, snippets, features) to (batch, features, snippets) to meet Conv1D input shape requirement
        x = x.permute(0, 2, 1)

        # Pass through the first Conv1D layer, activation and dropout
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # If use_second_conv is true, pass through the second Conv1D layer, activation and dropout
        if self.use_second_conv:
            x = self.conv2(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Change x shape back to (batch, snippets, hidden_dim) for the next layer
        x = x.permute(0, 2, 1)

        return x


# Convert each snippet feature vector into a single magnitude score using the L2 norm
def compute_feature_magnitude(snippet_features):

    feature_magnitudes = torch.norm(snippet_features, p=2, dim=2)

    return feature_magnitudes


# Select the top-k snippets with the highest magnitude scores for each video in the batch, default at 3
def topk_snippets_magnitude(feature_magnitudes, k=3):

    topk_values, topk_indices = torch.topk(feature_magnitudes, k=k, dim=1)

    return topk_values, topk_indices


# Collect the features of the top-k snippets based on the indices obtained
def topk_features(snippet_features, topk_indices):

    topk_features = torch.gather(
        snippet_features,
        dim=1,
        index=topk_indices.unsqueeze(-1).expand(-1, -1, snippet_features.size(2))
    )

    return topk_features


# Convert feature into equal segments for training, default at 32 segments
def convert_feature_equal_segment(feat, target_segment=32):

    # Create an empty array with 32 segments and same feature dimension to store the converted feature
    converted_feat = np.zeros((target_segment, feat.shape[1]), dtype=np.float32)

    # Create the segment/binning boundaries depending on the feature length and target segment count
    bin_edges = np.linspace(0, len(feat), target_segment + 1, dtype=int)

    # For each segment, calculate the average of the snippets' features values that fall within the segment and store it in the new feature array
    for i in range(target_segment):

        start, end = bin_edges[i], bin_edges[i + 1]

        if start != end:
            converted_feat[i, :] = np.mean(feat[start:end, :], axis=0)

        # This is to handle cases where the feature length is lesser than the target segment count.
        # In this case, the empty segment will fill with start index snippet's feature value
        else:
            converted_feat[i, :] = feat[start, :]

    return converted_feat


# Define a fully connected neural network model to learn feature vector of the top-k snippets and output a single anomaly score (logit) for each top-k snippets.
class anomaly_score_nn(nn.Module):

    def __init__(self, hidden_dim=512, dropout=0.5):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, top-k snippets, hidden_dim)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Combine Conv1D block, feature magnitude, top-k selection, and fully connected neural network model as a pipeline.
class anomaly_detection_pipeline(nn.Module):

    # input_dim is the I3D feature dimension
    # hidden_dim is the Conv1D output dimension
    # k is the number of top snippets to keep per video
    def __init__(self, input_dim=2048, hidden_dim=512, k=3, dropout=0.5, use_second_conv=True):

        super().__init__()

        self.k = k

        # Conv1D block for temporal feature learning across snippets
        self.conv_block = py_conv1d_block(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_second_conv=use_second_conv
        )

        # Anomaly scoring
        self.anomaly_score = anomaly_score_nn(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x):
        # x shape: (batch, snippets, features)

        # 1. Conv1D temporal feature learning
        snippet_features = self.conv_block(x)

        # 2. Compute each snippet feature magnitude
        feat_magnitude = compute_feature_magnitude(snippet_features)

        # 3. Select the top-k snippets by magnitude
        topk_values, topk_indices = topk_snippets_magnitude(feat_magnitude, k=self.k)

        # 4. Select top-k snippets features
        topk_feat = topk_features(snippet_features, topk_indices)

        # 5. Anomaly score on top-k features, then average over k for video-level anomaly score
        snippet_anomaly_score = self.anomaly_score(topk_feat).squeeze(-1)
        video_anomaly_score = snippet_anomaly_score.mean(dim=1, keepdim=True)

        # 6. Anomaly score for all snippets
        all_snippet_anomaly_score = self.anomaly_score(snippet_features).squeeze(-1)

        return video_anomaly_score, all_snippet_anomaly_score, topk_values


# Load the trained VAD model weights
def load_vad_model(weight_path):

    # Reload config and model state_dict from the saved checkpoint, then recreate the model architecture and load the weights.
    checkpoint = torch.load(weight_path, map_location="cpu")
    config = checkpoint["config"]

    # Recreate the model architecture with the same hyperparameters used during training, then load the trained weights into the model.
    model = anomaly_detection_pipeline(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        k=config["k"],
        dropout=config["dropout"],
        use_second_conv=config["use_second_conv"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)

    return model, checkpoint


# --- I3D feature load ---
# Load a saved I3D feature file and average the 10 crops, same as training.
def load_i3d_feature(feature_path):

    raw_i3d_feature = np.load(feature_path)
    snippet_features = raw_i3d_feature.mean(axis=1)

    return raw_i3d_feature, snippet_features.astype(np.float32)


# --- Anomaly Detection ---
# Run video anomaly detection from an existing I3D feature file.
def VAD(feature_path, threshold=0.2425, target_segment=32):

    raw_i3d_feature, snippet_features = load_i3d_feature(feature_path)

    # Convert I3D snippet features into the 32 equal segments used during training.
    vad_features = convert_feature_equal_segment(snippet_features, target_segment)

    vad_model, checkpoint = load_vad_model(vad_weight)

    # VAD model expects [batch, snippets, features].
    vad_input = torch.tensor(vad_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        video_anomaly_score, all_snippet_anomaly_score, topk_values = vad_model(vad_input)
        video_anomaly_prob = torch.sigmoid(video_anomaly_score).squeeze().cpu().item()
        all_snippet_anomaly_prob = torch.sigmoid(all_snippet_anomaly_score).squeeze(0).cpu().numpy()

    # To conclude whether the video contains anomaly scenes
    is_video_anomaly = video_anomaly_prob >= threshold

    # Same frame count estimate used during test evaluation in training.
    frame_count = raw_i3d_feature.shape[0] * 16

    # To store anomaly frame ranges
    anomaly_frame_ranges = []

    # To map the 32 snippet probabilities back to frame-level probabilities
    frame_anomaly_probs = np.zeros(frame_count, dtype=np.float32)
    vad_snippet_edges = np.linspace(0, frame_count, target_segment + 1, dtype=int)

    for snippet_index in range(target_segment):
        start = vad_snippet_edges[snippet_index]
        end = vad_snippet_edges[snippet_index + 1]
        frame_anomaly_probs[start:end] = all_snippet_anomaly_prob[snippet_index]

    detected_anomaly_frames = np.where(frame_anomaly_probs >= 0.5)[0]

    # Group individual anomaly frames into continuous ranges
    if len(detected_anomaly_frames) > 0:
        start_frame = detected_anomaly_frames[0]
        previous_frame = detected_anomaly_frames[0]

        for frame_index in detected_anomaly_frames[1:]:
            if frame_index == previous_frame + 1:
                previous_frame = frame_index
            else:
                anomaly_frame_ranges.append((start_frame, previous_frame))
                start_frame = frame_index
                previous_frame = frame_index

        anomaly_frame_ranges.append((start_frame, previous_frame))

    result = {
        "is_video_anomaly": is_video_anomaly,
        "video_anomaly_prob": video_anomaly_prob,
        "frame_anomaly_probs": frame_anomaly_probs,
        "anomaly_frame_ranges": anomaly_frame_ranges,
    }

    return result
