# Dynamic Memory Network Plus

This is the Pytorch implementation of the paper [Dynamic Memory Network for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417). This paper is an improved version of the original paper [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/pdf/1506.07285.pdf). The major difference between these ideas is in the functioning of the input module and the memory module which has been explained in detail in the IPython notebook file of this repo.

![Input Module for DMNPlus](https://raw.githubusercontent.com/hardik2396/Dynamic-Memory-network-plus/master/inputModule.png?token=AOUtTAtTVniqEEuulNufBGDcuXUTSG5Qks5bGvMewA%3D%3D)

## Description
- The whole architecture of DMN+ consists of 4 modules: Input Module, Memory Module, Question Module & the Answer Module.
- The input module uses Positional Encoder and BidirectionalGRU to encode the input text representation in a much better way than DMN.
- The memory module uses Attention based GRU to compute the contexual vector representing the input relevant to previous memory state and the question and finally uses this to update its next memory state.
- The question module uses a simple GRU to encode the question to get its vector representation.
- The answer module predicts the answer based on the final memory state and the question.
- CrossEntropyLoss has been used in the network and Adam optimizer to optimize the model parameters.
- The model has been trained on bAbI dataset which consists of 20 different question answering tasks.

## Requirements
  * Python 3.6
  * Pytorch
## Download Dataset
  ```
  * chmod +x fetch_data.sh
  * ./fetch_data.sh
 ```
## Usage
 Run the main python code
 ```
 python train_test.py
 ```

## References
- [Dynamic Memory Network for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417)
- [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/pdf/1506.07285.pdf)
- [https://github.com/dandelin/Dynamic-memory-networks-plus-Pytorch](https://github.com/dandelin/Dynamic-memory-networks-plus-Pytorch)
