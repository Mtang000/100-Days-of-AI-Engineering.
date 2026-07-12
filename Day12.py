import torch
import torch.nn as nn
import torch.optim as optim

users = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
videos = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
ratings = torch.tensor([5.0, 4.0, 5.0, 5.0, 4.0], dtype=torch.float32)


class AIRecommender(nn.Module):
    def __init__(self, num_users, num_videos, embedding_size=4):
        super().__init__()
        self.user_embed = nn.Embedding(
            num_embeddings=num_users, embedding_dim=embedding_size)
        self.video_embed = nn.Embedding(
            num_embeddings=num_videos, embedding_dim=embedding_size)

    def forward(self, user_idx, video_idx):
        user_profile = self.user_embed(user_idx)
        video_profile = self.video_embed(video_idx)

        prediction = (user_profile * video_profile).sum(dim=1)
        return prediction


model = AIRecommender(num_users=3, num_videos=4, embedding_size=4)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

epochs = 200
for epoch in range(epochs):
    predictions = model(users, videos)
    loss = criterion(predictions, ratings)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 40 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")


user_2_tensor = torch.tensor([2, 2], dtype=torch.long)
test_videos = torch.tensor([1, 3], dtype=torch.long)

with torch.no_grad():
    predicted_ratings = model(user_2_tensor, test_videos)

print("--- Predictions for User 2 ---")
print(
    f"Predicted rating for Video 1 (Gaming): {predicted_ratings[0].item():.1f} / 5.0 Stars")
print(
    f"Predicted rating for Video 3 (Coding): {predicted_ratings[1].item():.1f} / 5.0 Stars")
