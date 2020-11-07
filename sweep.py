# Init wandb

import wandb
wandb.init(project="con-rl")

# Log metrics with wandb
for _ in range(num_epochs):
    train_model()
    loss = calulate_loss()
    wandb.log({"Loss": loss})

# Save model to wandb

np.save("weights", weights)
wandb.save("weights.npy")