# BiRNN Transformer for Blood Glucose Prediction
## Dependencies
This repository works on Python3.8.8

## Proprocessing
Acquire the raw OhioT1DM data from http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html.

The data should have the following file structure:
```
OhioT1DM
|-- OhioT1DM-training
|-- OhioT1DM-testing
```
Then run the following command to preprocess the data using our scripts.
```
python3 ./preprocess/linker.py --data_folder_path path/to/ohiot1dm --extract_folder_path ./data
```

## Train
An example of replicating the setting 1 on subject 540 for 30 minutes prediciton horizon. 
```
python3 train.py --patient 540 --missing_len 6 --single_pred --transfer_learning
```
An example of replicating the setting 2 on subject 540 for 30 minutes prediciton horizon. 
```
python3 train.py --patient 540 --missing_len 6 --transfer_learning
```
An example of replicating the ablation study 1 on subject 540 for 30 minutes prediciton horizon. 
```
python3 train.py --patient 540 --missing_len 6 --single_pred --unimodal --transfer_learning
```
An example of replicating the ablation study 2 on subject 540 for 30 minutes prediciton horizon. 
```
python3 train.py --patient 540 --missing_len 6 --single_pred
python3 train.py --patient 540 --missing_len 6
```

## License
MIT
