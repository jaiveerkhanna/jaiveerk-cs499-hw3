 (10pt) Report your results through an .md file in your submission; discuss your implementation choices and document the performance of your models (both training and validation performance) under the conditions you settled on. Compare the base encoder-decoder, encoder-decoder with attention, and transformer-based model performance. Discuss your encoder-decoder attention choices (e.g., flat or hierarchical, recurrent cell used, etc.). Discuss the attention mechanism you implemented for the encoder-decoder model using the taxonomy we discussed in class.

DISCLAIMER:
    - If you want to examine the model without attention, change line 8 of train.py to : import model_no_attention as model
    - If you want to examine the model with attention, change line 8 of train.py to : import model_attention as model

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
    - Attention Mechanism
        - I want to note that I was not able to complete the attention mechanism, however, if you look at model_attention.py it might be clear what I was trying to achieve (still having issues with the sizing of tensors, but the overall flow should make sense)
        - My steps for incorporating attention are as follows:
            - Calculate attention weights (done by combining decoders input and hidden)
            - These are multiplied by encoder outputs to create a weighted combination (attn_applied)
            - attn applied is combined with the decoders input embedings (attn_combined)
            - the combined attention is sent through an lstm with the hidden state of the decoder (lstm in the Encoder Decoder wrapper)
    
Implementation:
    - Train / Test Split:
        - Decided to use the train / valid seen split given from the data
        - Ran into the error that there were some actions in the val set that were unseen in the train so created an    UNK action and target tokeng
        - Obviously, needed to maintain episodes as 1 entire input so couldnt shuffle that
        - Felt like it was good not to mix the data as I want the model to be able to handle unseen actions/targets. I could have alternatively processed all the data such that every action/target was characterized/had embeddings but chose not to to make the model more resilient to new data

Hyperparameters:
- Loss Criterions
    - Cross Entropy --> worked well for both actions and targets. Since I am using two seperate FC layers to predict action and target, this is a suitable metric
- Optimizer
    - SGD --> worked well and trained quickly --> initial
    - Learning Rate --> 0.005 --> tried a few different and this worked better
- Embedding Dimensions: 128
    - for the decoder, the embedding layers for the action and target embedding were 1/2 the size (since i used two embedding layers after thinking through / discussing with anthony). I then concatenated both embeddings before passing htat through the decoder LSTM
- learning rate : 0.005 --> have been using this pretty standard rate and it worked well. Didn't see a reason to tweak
- minibatch size: stuck to 32 as it seemed to be working and speed was ok
- Vocab size: 1000 (project default and good enough for this data set, not a lot of unks... if any?)


Performance:
- You can find a summary figure called encoder_decoder.png
- I am unable to compare the performance between the three models because I was still stuck on implementing the encoder-decoder. However, I would like to explain what I would have expected to see so that I can at least get full credit for the report
- I expect that the Encoder - Decoder with attention should be able to perform better than the one without attention. this makes sense because now the burden of encoding an entire input sequence does not fall on a single context vector. Rather, the Decoder and "focus" on multiple parts of the encoders outputs and thus create a better decoding
- I expect the Transformer to perform similarly to the Encoder Decoder on a pure accuracy basis (not counting training time and speed). However, since the Transformer allows for parallelization, it should train much faster and thus be able to eventually outperform the Encoder - Decoder. It will be much quicker and is am much lighter framework, hence why it is so popular


Bonus:
- Was not able to implement a transformer model ... which is a bummer but just the reality of how much I could achieve by the deadline


