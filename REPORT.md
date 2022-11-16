 (10pt) Report your results through an .md file in your submission; discuss your implementation choices and document the performance of your models (both training and validation performance) under the conditions you settled on. Compare the base encoder-decoder, encoder-decoder with attention, and transformer-based model performance. Discuss your encoder-decoder attention choices (e.g., flat or hierarchical, recurrent cell used, etc.). Discuss the attention mechanism you implemented for the encoder-decoder model using the taxonomy we discussed in class.

 Architecture:
    -Encoder Decoder Wrapper Model does the following:
        - Takes input, runs through encoder
        - Encoder will take input, create embeddings, run LSTM, and return lstm final state
        - Encoder/Decoder takes the hidden state of encoder and sets that as the first hidden state of the decoder
        - Encoder/Decoder then loops:
            - For each instruction, pass Hidden Decoder State to 2 FC layers (action and target)
            - Feed the resultant action/target pair as well as the hidden state of the decoder to the decoder LSTM
            - Save the returned decoder LSTM hidden state and repeat
        - Take all action/target pairs and return that to main

Hyperparameters:
- Loss Criterions
    - Cross Entropy --> worked well for both actions and targets. Since I am using two seperate FC layers to predict action and target, this is a suitable metric
- Optimizer
    - SGD --> worked well and trained quickly
    - Learning Rate --> 0.005 --> tried a few different and this worked better
- Embedding Dimensions: 128
    - for the decoder, the embedding layers for the action and target embedding were 1/2 the size (since i used two embedding layers after thinking through / discussing with anthony). I then concatenated both embeddings before passing htat through the decoder LSTM
- learning rate : 0.005 --> have been using this pretty standard rate and it worked well. Didn't see a reason to tweak
    
    