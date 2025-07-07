# Deep Learning with PyTorch 🚀

Welcome to **Deep Learning with PyTorch** — a practical and modular codebase designed for learning and experimenting with deep learning using Python and PyTorch.

This repository contains clean, well-organized `.py` scripts for:
- Learning core PyTorch concepts
- Building and training neural networks
- Running experiments on popular datasets
- Extending to custom projects

All code is written for use with **Visual Studio Code**, Python 3.x, and the PyTorch ecosystem.

---

## 📂 Project Structure

deep-learning-with-pytorch/
│
├── data/ # Datasets and loaders (if any)
├── models/ # Custom model definitions
├── utils/ # Helper functions (metrics, plotting, etc.)
├── experiments/ # Training scripts and experiments
├── checkpoints/ # Saved model weights
├── main.py # Entry point (optional)
├── requirements.txt
└── README.md


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/deep-learning-with-pytorch.git
cd deep-learning-with-pytorch
```
2. Set Up the Environment

Install the required dependencies:
```
pip install -r requirements.txt
```
3. Run a Script

Example: training a simple MLP on MNIST
```
python experiments/train_mnist.py
```
🧠 What's Inside
🔧 Core Features

    Modular scripts for easy reuse

    Clean separation of model, data, and training logic

    Logging and checkpointing support

    Extensible for your own datasets and models

📚 Covered Topics

    Tensors and operations

    Forward and backward passes

    Model training and evaluation loops

    Optimizers, losses, and learning rate schedulers

    CNNs, MLPs, and transfer learning (WIP)

🛠️ Requirements

Make sure you have Python 3.8+ and PyTorch installed.
```
torch
torchvision
numpy
matplotlib
pandas
scikit-learn
```
VS Code users: you can use the provided .vscode/settings.json (if included) for a pre-configured linting/debug setup.

🤝 Contributing

Feel free to open issues or submit pull requests. Whether it’s bug fixes, new models, or better training utilities, contributions are welcome!
📜 License

This repository is licensed under the MIT License.
📬 Contact

Maintained by broshann14@gmail.com
Questions? Feedback? Open an issue or drop a message.
