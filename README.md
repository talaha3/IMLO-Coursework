# Python Script and CNN Model for 2023/2024 IMLO Individual Assessment

This project implements a Convolutional Neural Network for classifying flower images from the Oxford Flowers-102 dataset using PyTorch.

It is strongly recommended to run the training program on a computer with a GPU.

## Set up

1. **Clone repository**:

   ```sh
   git clone #hidden for anonymity#
   ```

2. **Create the Python virtual environment**:

   ```sh
   cd IMLOAssessment
   python -m venv myenv
   ```

3. **Activating virtual environment**:

   #### On Windows:

   ```sh
   ./myenv/Scripts/activate.bat
   ```

   #### On Mac and Linux:

   ```sh
   $ source myvenv/bin/activate
   ```

4. **Install the required packages**:

   ```sh
   pip install torch torchvision matplotlib scipy
   ```

## Usage

### Training the model using training dataset

```sh
python finalClassifier.py
```

### Testing the train model on test dataset

```sh
python modelTest.py
```

## Files

`/attempts`: Contains previous attempts at creating classifier.

`finalClassifier.py`: Trains the model and saves most accurate parameter configuration.

`modelTest.py`: Loads saved model and tests accuracy on the test dataset.

`bestmodel.pt`: Saved model with trained parameters yielding highest accuracy.

`dataset_mean_std.py`: Calculates and prints the mean and standard deviation of the training set.
