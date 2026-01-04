
import matplotlib.pyplot as plt

from trl import SFTTrainer


def plot_training_loss(trainer: SFTTrainer):
    # Access the log history
    log_history = trainer.state.log_history

    # Extract training / validation loss
    train_losses = [log["loss"] for log in log_history if "loss" in log]
    epoch_train = [log["epoch"] for log in log_history if "loss" in log]
    eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]

    # Plot the training loss
    plt.plot(epoch_train, train_losses, label="Training Loss")
    plt.plot(epoch_eval, eval_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()