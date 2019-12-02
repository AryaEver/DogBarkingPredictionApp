# Dog Barking Prediction App

In this project there are 3 stages to follow from step to step.

<img src="./overall procedure.png" height="250" />

**1- Pre-processing:** 

in this stage user should download the provided datasets from the following link "[not yet hosted]" or add manually from a local source and put them in the dataset folders. Then run the python script in the 1-preprocessing folder to rename and reorganize all the files that are in the same category and exclude exceptions that are not labeled. The previous procedure should generate a .csv file containing all files organized by each classification task (name, breed, age, sex, context) with all the matching labeled audiofiles information and their labels in the last column duration should be at max 1 sec and empty space will be filled with zeros also the sample rate now its at 8820Hz, so the number of columns in th .csv file is 8820 data + 1 for labels and rows are the number of audio files.  

**2- Characterization:**

in this stage user should proceed to evaluate and select which characterization techinques to use options provided are: 
+ Proceed with raw audio (Data augmentation should be done here if planned to use). 
+ Compute Melspectrograms over raw audio using librosa in a python script.
+ Extract a group of low level descriptors with external software (OpenSmile).
+ "[not yet implmented]"Extract Specific group of low level descriptors using librosa in a python script.
Then eduction of dimension of size of feature set.

**3- Classifying:**  

**Additionally:** 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the following packages.

```bash
pip install tensorflow-gpu 
pip install keras
pip install pandas
pip install librosa
pip install seaborn
pip install scikit-learn
```

## Usage

## Contributing
If you have any trouble or instructions are not clear, please comment the issue so i can fix it or explain it with more detail.
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Opensource
