# HOI4 Self-Learning AI 🧠🎮

An autonomous AI system that learns to play Hearts of Iron 4 through pure self-play and exploration, without any human demonstrations or hardcoded strategies.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### Ultimate AI Mode (Recommended)
- **DreamerV3-Inspired World Model**: Learns game dynamics through imagination
- **Random Network Distillation (RND)**: Curiosity-driven exploration of game mechanics
- **Neural Episodic Control (NEC)**: Lightning-fast learning from single experiences
- **Persistent Memory**: Remembers strategies across game sessions using ChromaDB

### Core Capabilities
- **Pure Self-Play Learning**: No human demonstrations required
- **Causal Understanding**: Discovers cause-and-effect relationships
- **Strategic Memory**: "I remember from 3 games ago..." moments
- **Multi-Modal Perception**: OCR + Vision + Game State Understanding

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Windows 10/11 (for pyautogui compatibility)
- Hearts of Iron 4 (any version)
- NVIDIA GPU recommended (but not required)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hoi4-ai.git
cd hoi4-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install chromadb faiss-cpu einops tensordict
```

3. Install Tesseract OCR:
- Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Set TESSERACT_PATH environment variable or install to default location

### Running the AI

1. Start HOI4 in **windowed mode** (not fullscreen)
2. Load a game as Germany 1936
3. Pause the game (spacebar in HOI4)
4. Run the AI:
```bash
python main.py --mode ultimate
```
5. Press **F5** to start AI learning

### Controls
- **F5**: Start/Resume AI
- **F6**: Pause AI
- **F7**: Save Progress
- **F8**: Show Statistics
- **ESC** (hold 2s): Stop AI

## 🎯 AI Modes

### Ultimate Mode (Default)
Combines all cutting-edge technologies:
- World model-based planning
- Curiosity-driven exploration
- Fast episodic learning
- Cross-game memory

### Other Modes
- `--mode strategic`: Pure reinforcement learning
- `--mode understanding`: Focus on game comprehension
- `--mode integrated`: Combines understanding with strategy
- `--mode record`: Record your gameplay for analysis

## 📊 What to Expect

### Timeline
- **First 5 minutes**: Random exploration, discovering UI elements
- **After 10 minutes**: Pattern recognition begins
- **After 30 minutes**: Develops opening strategies
- **After 1 hour**: Consistent strategic play
- **After multiple games**: Cross-game learning and optimization

### Milestones to Watch For
- 🎉 "Discovered new screen: production" - Finding game menus
- 💡 "I remember: Game 1: Built factory..." - Using past experience
- 🔍 "High curiosity (2.34) - exploring!" - Discovering new mechanics
- 🏭 "Factory growth: +1" - Successfully building infrastructure

## 🏗️ Architecture

### Core Components

#### 1. Perception Layer (`src/perception/`)
- **OCR Engine**: Reads game text and numbers
- **Screen Analysis**: Understands current game screen

#### 2. Comprehension System (`src/comprehension/`)
- **Understanding Engine**: Builds mental model of game mechanics
- **Curiosity System**: Drives exploration of unknown features
- **Language Parser**: Interprets game UI text

#### 3. Learning Systems (`src/ai/`)
- **World Model**: Predicts future game states
- **RND Curiosity**: Intrinsic motivation system
- **Neural Episodic Control**: Fast memory-based learning
- **Strategic Evaluator**: Assesses progress toward victory

#### 4. Ultimate AI (`src/ai/ultimate/`)
- **Integrated System**: Combines all components
- **Persistent Memory**: Cross-game learning with ChromaDB
- **Training Loop**: Manages the learning process

## 🔧 Configuration

### Performance Tuning
```python
# In src/ai/ultimate/ultimate_ai.py
memory_size = 50000        # Reduce if low on RAM
replay_buffer_size = 100000  # Reduce if low on RAM
device = 'cuda'           # Change to 'cpu' if no GPU
```

### Screen Resolution
The AI automatically adapts to any screen resolution. No configuration needed!

### Game Speed
The AI controls game speed automatically. Start with the game paused.

## 📈 Monitoring Progress

### Statistics (F8)
- Total steps taken
- Average intrinsic reward
- Memory utilization
- Discoveries made

### Saved Files
- `checkpoints/`: Model checkpoints every 5 minutes
- `hoi4_persistent_memory/`: Cross-game memories
- `models/`: Trained model files

## 🐛 Troubleshooting

### Common Issues

**OCR not reading text**
- Ensure Tesseract is installed and in PATH
- Check if `TESSERACT_PATH` environment variable is set
- Try running `tesseract --version` in terminal

**AI clicking too fast/slow**
- Adjust sleep time in `ultimate_ai.py` (default 0.1s)
- Check if game is running at normal speed

**High memory usage**
- Reduce `memory_size` in `NeuralEpisodicControl`
- Reduce `replay_buffer` capacity
- Use CPU instead of GPU

**CUDA out of memory**
- Set `device='cpu'` in `ultimate_ai.py`
- Reduce batch sizes
- Close other GPU applications

### Debug Mode
```bash
python main.py --mode ultimate --debug
```

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Multi-country support (currently optimized for Germany)
- Multiplayer compatibility
- Performance optimizations
- Additional game mechanics understanding

## 📚 Technical Details

### Learning Algorithm
The Ultimate AI uses a combination of:
1. **Model-Based RL**: DreamerV3-style world model
2. **Intrinsic Motivation**: RND for exploration
3. **Episodic Control**: Fast learning from memories
4. **Causal Discovery**: Understanding game mechanics

### Key Innovations
- Resolution-independent action space
- Dynamic curiosity adjustment
- Persistent cross-game memory
- Causal relationship discovery

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- DreamerV3 paper (Hafner et al., 2023)
- Random Network Distillation (Burda et al., 2018)
- Neural Episodic Control (Pritzel et al., 2017)
- The HOI4 modding community

## 📧 Contact

For questions or collaboration: [your-email@example.com]

---

*"The AI that learns to conquer the world, one click at a time."* 🌍
