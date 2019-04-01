# State Frequency Memory recurrent network for stock price prediction
Author: Liheng Zhang, Date: 08/03/2017

This is the project for the following paper:
    
    Liheng Zhang, Charu Aggarwal, Guo-Jun Qi, Stock Price Prediction via Discovering Multi-Frequency Trading Patterns,
    in Proceedings of ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2017), Halifax, Nova Scotia,
    Canada, August 13-17, 2017.
    
Questions about the source codes can be directed to Liheng Zhang at lihengzhang1993@knights.ucf.edu.

For more applications with SFM, please refer to:

    Hao Hu, Guo-Jun Qi. State-Frequency Memory Recurrent Neural Networks, in Proceedings of International Conference
    on Machine Learning (ICML 2017), Sydney, Australia, August 6-11, 2017.
    
## Requirements
- Python == 2.7
- Keras == 1.0.1
- Theano == 0.9

## Prepare the data
    cd dataset; python build_data.py
    
## Test with pretrained model
    cd test
    python test.py --step=1
The model for n-step prediction is specified with --step. Models for 1-step, 3-step and 5-step prediction are provided.

To visualize the predicted results:

    python test --step=1 --visualization=true
    
## Training
    cd train
    python train.py --step=3 --hidden_dim=50 --freq_dim=10 --niter=4000 --learning_rate=0.01
    
## Note
The codes are expired for Keras >= 2.0.0.

    

