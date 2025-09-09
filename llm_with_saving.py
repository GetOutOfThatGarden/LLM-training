import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pickle
import os

class SimpleLLM:
    def __init__(self, model_name="simple_llm"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.max_sequence_length = None
        self.total_words = None
        
    def prepare_data(self, data):
        """Tokenize and prepare sequence data"""
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(data)
        self.total_words = len(self.tokenizer.word_index) + 1
        
        # Create input sequences
        input_sequences = []
        for line in data:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        
        self.max_sequence_length = max([len(x) for x in input_sequences])
        input_sequences = pad_sequences(input_sequences, maxlen=self.max_sequence_length, padding='pre')
        
        return input_sequences
    
    def build_model(self):
        """Build the LLM architecture"""
        self.model = Sequential()
        self.model.add(Embedding(self.total_words, 100, input_length=self.max_sequence_length-1))
        self.model.add(LSTM(150, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100))
        self.model.add(Dense(self.total_words, activation='softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def train_with_checkpoints(self, data, epochs=200, checkpoint_freq=10, resume_training=True):
        """Train model with automatic checkpointing"""
        
        # Prepare data
        input_sequences = self.prepare_data(data)
        X, y = input_sequences[:, :-1], input_sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=self.total_words)
        
        # Create directories for saving
        checkpoint_dir = f"checkpoints/{self.model_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Try to load existing model and tokenizer
        start_epoch = 0
        if resume_training and self.load_checkpoint():
            print(f"Resumed training from existing checkpoint")
            start_epoch = self.get_last_epoch()
        else:
            print("Starting training from scratch")
            self.build_model()
        
        # Save tokenizer and metadata
        self.save_tokenizer_and_metadata()
        
        # Set up callbacks
        callbacks = [
            # Save model every checkpoint_freq epochs
            ModelCheckpoint(
                filepath=f"{checkpoint_dir}/model_epoch_{{epoch:03d}}.h5",
                save_freq=checkpoint_freq * len(X) // 32,  # Assuming batch_size=32
                save_weights_only=False,
                verbose=1
            ),
            # Save best model based on loss
            ModelCheckpoint(
                filepath=f"{checkpoint_dir}/best_model.h5",
                monitor='loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train the model
        remaining_epochs = epochs - start_epoch
        if remaining_epochs > 0:
            history = self.model.fit(
                X, y, 
                epochs=remaining_epochs,
                initial_epoch=start_epoch,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save final model
            self.save_final_model()
            return history
        else:
            print("Model already trained for requested epochs")
            return None
    
    def save_tokenizer_and_metadata(self):
        """Save tokenizer and training metadata"""
        checkpoint_dir = f"checkpoints/{self.model_name}"
        
        # Save tokenizer
        with open(f"{checkpoint_dir}/tokenizer.pickle", 'wb') as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata
        metadata = {
            'total_words': self.total_words,
            'max_sequence_length': self.max_sequence_length
        }
        with open(f"{checkpoint_dir}/metadata.pickle", 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_checkpoint(self, epoch=None):
        """Load model from checkpoint"""
        checkpoint_dir = f"checkpoints/{self.model_name}"
        
        try:
            # Load tokenizer and metadata
            with open(f"{checkpoint_dir}/tokenizer.pickle", 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            with open(f"{checkpoint_dir}/metadata.pickle", 'rb') as f:
                metadata = pickle.load(f)
                self.total_words = metadata['total_words']
                self.max_sequence_length = metadata['max_sequence_length']
            
            # Load model
            if epoch is not None:
                model_path = f"{checkpoint_dir}/model_epoch_{epoch:03d}.h5"
            else:
                # Load the best model or most recent
                if os.path.exists(f"{checkpoint_dir}/best_model.h5"):
                    model_path = f"{checkpoint_dir}/best_model.h5"
                elif os.path.exists(f"{checkpoint_dir}/final_model.h5"):
                    model_path = f"{checkpoint_dir}/final_model.h5"
                else:
                    # Find the most recent epoch checkpoint
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_')]
                    if checkpoints:
                        latest = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
                        model_path = f"{checkpoint_dir}/{latest}"
                    else:
                        return False
            
            self.model = load_model(model_path)
            print(f"Loaded model from {model_path}")
            return True
            
        except (FileNotFoundError, EOFError) as e:
            print(f"Could not load checkpoint: {e}")
            return False
    
    def get_last_epoch(self):
        """Get the epoch number of the last saved checkpoint"""
        checkpoint_dir = f"checkpoints/{self.model_name}"
        try:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_')]
            if checkpoints:
                epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoints]
                return max(epochs)
            return 0
        except:
            return 0
    
    def save_final_model(self):
        """Save the final trained model"""
        checkpoint_dir = f"checkpoints/{self.model_name}"
        self.model.save(f"{checkpoint_dir}/final_model.h5")
        print(f"Final model saved to {checkpoint_dir}/final_model.h5")
    
    def generate_text(self, seed_text, next_words=15):
        """Generate text using the trained model"""
        if self.model is None or self.tokenizer is None:
            if not self.load_checkpoint():
                print("No trained model found. Please train the model first.")
                return None
        
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_length-1, padding='pre')
            predicted_probabilities = self.model.predict(token_list, verbose=0)[0]
            predicted_index = np.argmax(predicted_probabilities)
            
            if predicted_index in self.tokenizer.index_word:
                output_word = self.tokenizer.index_word[predicted_index]
                seed_text += " " + output_word
            else:
                break
                
        return seed_text

# Example usage
if __name__ == "__main__":
    # Sample data (replace with your actual dataset)
    data = [
        
        "Great Steeping is a village and civil parish in the East Lindsey district of Lincolnshire, England.",
"It is situated approximately 3 miles (5 km) from Spilsby.",
"The parish includes the hamlet of Monksthorpe.",
"There are two churches dedicated to All Saints, one being redundant and now known as Old All Saints.",
"Old All Saints, built in 1748 on the site of a medieval church, and restored in 1908, is a Grade II* listed building.",
"The Diocese of Lincoln declared it redundant in August 1973.",
"In the grounds is the socket stone of a medieval churchyard cross which is an ancient scheduled monument.",
"All Saints’ Church was built of red brick in 1891, after a design by William Bassett-Smith.",
"It is Grade II listed, and has a listed churchyard cross.",
"Great Steeping Primary School was built in 1859, and later run by the Great Steeping School Board from 1876 to 1903 as Great Steeping Board School.",
"Kelsey Hall dates from 1854 but occupies the site of an earlier manor house which burnt down.",
"It is first noted in 1507 as Kelsayhall and its name derives from the Kelsey family associated with Great Steeping.",
"Old documents refer to William de Kellessay in Steping in 1299, and Ralph de Kelsay in 1327. Kelsey Hall is a private house.",
"Great Steeping was also the base for RAF Spilsby, which originally was to be on the site of Gunby Park.",
"However, after an appeal by Field Marshal Sir Archibald Montgomery-Massingberd of Gunby Hall to the King, the RAF Steeping airfield was built as RAF Spilsby.",
"It opened in September 1943, and in 1944 RAF Spilsby, RAF Strubby, and RAF East Kirkby joined to become the newly formed 55 Base with headquarters at East Kirkby.",
"In September 1944 RAF Spilsby became a station for two Lancaster squadrons, the 207 and 44.",
"It was taken over by No 2 Armament Practice School from 1945 until November 1946, after which the station was placed on care and maintenance until 1955.",
"It re-opened to host ground units of the USAF until they moved out in 1958."
        # Add more training data here...
    ]
    
    # Create LLM instance
    llm = SimpleLLM("fairy_tale_model")
    
    # Train with checkpointing (will resume if checkpoints exist)
    llm.train_with_checkpoints(data, epochs=200, checkpoint_freq=10)
    
    # Generate text
    generated = llm.generate_text("Once upon a time", next_words=20)
    print(f"Generated text: {generated}")
    
    # To load a specific epoch later:
    # llm.load_checkpoint(epoch=50)
    
    # To continue training with more epochs:
    # llm.train_with_checkpoints(data, epochs=300)  # Will resume from where it left off


# Quick Start - Just replace the original training code with:
# llm = SimpleLLM("my_model")
# llm.train_with_checkpoints(data, epochs=200, checkpoint_freq=10)

# To resume interrupted training:
# llm = SimpleLLM("my_model")  # Same name as before
# llm.train_with_checkpoints(data, epochs=200)  # Automatically resumes

# To generate text later:
# llm = SimpleLLM("my_model")
# text = llm.generate_text("Once upon a time", next_words=15)