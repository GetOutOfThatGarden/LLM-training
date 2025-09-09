# Simple LLM Training Project
A TensorFlow-based implementation of a simple Language Learning Model (LLM) with automatic checkpointing and text generation capabilities.

## 📁 Project Structure
```
LLM-training/
├── llm_with_saving.py    # Main 
LLM implementation with 
checkpointing
├── simple_llm.py         # Basic 
LLM implementation
├── checkpoints/          # Model 
checkpoints and saved states
│   └── fairy_tale_model/ # Example 
trained model
└── venv/                 # Python 
virtual environment
```
## 🚀 Features
- Automatic Checkpointing : Save model progress every N epochs
- Resume Training : Continue training from the last checkpoint
- Text Generation : Generate text using trained models
- Model Persistence : Save and load trained models with metadata
- Early Stopping : Prevent overfitting with automatic early stopping
- Best Model Tracking : Automatically save the best performing model
## 🛠️ Installation
### Prerequisites
- Python 3.7+
- Virtual environment (recommended)
### Setup
1. 1.
   Clone or download the project
2. 2.
   Create and activate virtual environment :
   
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # 
   Linux/Mac
   ```
3. 3.
   Install dependencies :
   
   ```
   pip install tensorflow numpy
   ```
## 📖 Usage
### Basic Training
```
from llm_with_saving import 
SimpleLLM

# Prepare your training data
data = [
    "Your training text here",
    "More training sentences",
    "Add as many as needed"
]

# Create and train the model
llm = SimpleLLM("my_model")
llm.train_with_checkpoints(data, 
epochs=200, checkpoint_freq=10)
```
### Text Generation
```
# Generate text using trained model
generated = llm.generate_text("Once 
upon a time", next_words=20)
print(f"Generated text: {generated}
")
```
### Resume Training
```
# Resume training from existing 
checkpoint
llm = SimpleLLM("my_model")  # Same 
model name
llm.train_with_checkpoints(data, 
epochs=300)  # Will resume 
automatically
```
### Load Specific Checkpoint
```
# Load a specific epoch
llm = SimpleLLM("my_model")
llm.load_checkpoint(epoch=50)
text = llm.generate_text("Start 
text", next_words=15)
```
## 🏗️ Model Architecture
- Embedding Layer : 100-dimensional word embeddings
- LSTM Layers :
  - First LSTM: 150 units with return sequences
  - Second LSTM: 100 units
- Dropout : 0.2 rate for regularization
- Dense Output : Softmax activation for next word prediction
- Optimizer : Adam
- Loss Function : Categorical crossentropy
## 📊 Training Features
### Automatic Checkpointing
- Saves model every checkpoint_freq epochs
- Saves best model based on loss
- Saves final model after training completion
- Stores tokenizer and metadata
### Early Stopping
- Monitors training loss
- Stops training if no improvement for 20 epochs
- Restores best weights automatically
### File Structure After Training
```
checkpoints/
└── your_model_name/
    ├── best_model.h5           # 
    Best performing model
    ├── final_model.h5          # 
    Final trained model
    ├── model_epoch_XXX.h5      # 
    Periodic checkpoints
    ├── tokenizer.pickle        # 
    Trained tokenizer
    └── metadata.pickle         # 
    Model metadata
```
## 🔧 Configuration Options
Parameter Default Description model_name "simple_llm" Name for saving/loading models epochs 200 Number of training epochs checkpoint_freq 10 Save checkpoint every N epochs resume_training True Resume from existing checkpoint next_words 15 Number of words to generate

## 📝 Example Training Data
The project includes example training data about:

- Fairy tale beginnings
- Historical information about Great Steeping village
- Kitten development and care (in simple_llm.py)
## 🚨 Common Issues
### ModuleNotFoundError: No module named 'tensorflow'
Solution : Activate virtual environment and install TensorFlow:

```
venv\Scripts\activate
pip install tensorflow
```
### Empty Input Sequences Error
Solution : Ensure you have sufficient training data with multiple sentences.

### Memory Issues
Solution : Reduce batch size or use smaller model dimensions.

## 🤝 Contributing
1. 1.
   Fork the repository
2. 2.
   Create a feature branch
3. 3.
   Make your changes
4. 4.
   Test thoroughly
5. 5.
   Submit a pull request
## 📄 License
This project is open source and available under the MIT License.

## 🔮 Future Enhancements
- Support for different model architectures
- Hyperparameter tuning
- Evaluation metrics
- Data preprocessing utilities
- Web interface for text generation
- Support for larger datasets
Note : This is a simple educational implementation of an LLM. For production use, consider more sophisticated architectures like Transformers.