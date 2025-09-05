import matplotlib.pyplot as plt


def load(log_path):
    losses, lrs = [], []
    with open(log_path, 'r') as f:
        for line in f:
            splt = line.split()
            loss = splt[3]
            lr = splt[4]

            loss = loss[loss.find(':')+1:]
            lr = lr[lr.find(':')+1:]
            
            losses.append(float(loss))
            lrs.append(float(lr))

    return losses, lrs


if __name__ == "__main__":
    log_path = '/tmp/torchtune/llama3_1_8B/lora_single_device/logs/log_1743771284.txt'
    losses, lrs = load(log_path)

    plt.figure(figsize=(12, 6))
    plt.grid(True)
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.plot(list(range(len(losses))), losses)

    plt.subplot(1, 2, 2)
    plt.title('Learning Rate')
    plt.xlabel('step')
    plt.ylabel('lr')
    plt.plot(list(range(len(lrs))), lrs)

    plt.savefig('llama8.png')

