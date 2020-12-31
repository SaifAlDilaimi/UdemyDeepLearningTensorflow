learning_rate = 1e-3
epochs = 10
i = 0

def compute_lr(lr, epoch=0):
    new_lr = lr * (1 / (epoch+1))
    print("Current learning_rate ", new_lr)

    if (epoch+1) != epochs:
        compute_lr(new_lr, epoch=epoch+1)


print("Starting with learning_rate ", learning_rate)
compute_lr(learning_rate)