:: python train.py -add_noise False -pe onehot   -id OH_Bias-Tanh_0.00
:: python train.py -add_noise False -pe None


python test.py -add_noise -noise_level 0.00 -pe onehot -id OH_Bias-Tanh_0.00
python test.py -add_noise -noise_level 0.05 -pe onehot -id OH_Bias-Tanh_0.05
python test.py -add_noise -noise_level 0.00 -pe sincos -id SC_Bias-Tanh_0.00
python test.py -add_noise -noise_level 0.05 -pe sincos -id SC_Bias-Tanh_0.05
python test.py -add_noise -noise_level 0.00 -pe relative -id RE_Bias-Tanh_0.00
python test.py -add_noise -noise_level 0.05 -pe relative -id RE_Bias-Tanh_0.05
python test.py -add_noise -noise_level 0.00 -id NO_Bias-Tanh_0.00
python test.py -add_noise -noise_level 0.05 -id NO_Bias-Tanh_0.05
python test.py -add_noise -noise_level 0.00 -pe onehot -activ tanh -id OH_Tanh_0.00
python test.py -add_noise -noise_level 0.05 -pe onehot -activ tanh -id OH_Tanh_0.05
python test.py -add_noise -noise_level 0.00 -pe onehot -activ sigmoid -id OH_Sigmoid_0.00
python test.py -add_noise -noise_level 0.05 -pe onehot -activ sigmoid -id OH_Sigmoid_0.05
python test.py -add_noise -noise_level 0.00 -pe onehot -activ relu -id OH_ReLU_0.00
python test.py -add_noise -noise_level 0.05 -pe onehot -activ relu -id OH_ReLU_0.05
python test.py -add_noise -noise_level 0.00 -pe onehot -activ None -id OH_NO_0.00
python test.py -add_noise -noise_level 0.05 -pe onehot -activ None -id OH_NO_0.05
