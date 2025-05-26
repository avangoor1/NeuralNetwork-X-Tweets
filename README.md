# Neural Network Training Guide

Now that you have collected tweets, let's explore the steps to train a neural network.

## Step 0: Set up a Virtual Environment

Setting up a virtual environment ensures that all required libraries and their specific versions are installed in an isolated space, preventing conflicts with other projects on your system.

Create and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 1: Install Dependencies

There is a file called `requirements.txt` that contains all the libraries and their specific versions needed to successfully run the files in this repository.

Install the required libraries:

```bash
pip install -r requirements.txt
```

**Your system is now ready to train a neural network. However, depending on the power of your machine, the training process may take a long time to complete. You can also explore other options like using Google Colab for training purposes.**

## Step 2: Train Your Neural Network

Train your neural network by running the `crossFoldValidation.py` program. 

You must split your collected tweets into five folds of training and testing sets. You can explore an example of these sets in the folder titled "Cross Fold Validation Sets". 

**Important:** Edit the program to add the folder that contains your training and testing sets.

When you run the program, it will automatically store the generated model files along with their accuracy metrics in their respective folders. The program will output five different model files for you to use, but you should use the one with the highest overall accuracy.

## Step 3: Expand Test Tweets

Once you have collected tweets from the specific group you want to test your model on, the best classification results are achieved when the test set tweets are expanded.

To expand your tweets, use the `OpenAPI.py` program and pass each tweet you want to expand through it.

## Step 4: Use Your Trained Model

Now that you have a trained model and a set of tweets from a specific community, you can analyze where these tweets fall on the spectrum of luck versus meritocracy by evaluating them using the neural network. Use the `model_pred.py` script to assist with this analysis.

---


**Note:** For optimal performance and faster training times, consider using cloud-based solutions like Google Colab, especially if you have limited local computing resources.