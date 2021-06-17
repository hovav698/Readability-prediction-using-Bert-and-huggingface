This repo will contain PyTorch implementation of BERT model for solving text readability score. 
It's aimed at making it easy to use BERT model and NLP methods for text classification and predictions. 
I will use a library called huggingface, which make it easier to use and implement NLP state of the art models.

**The data and the goal**

The dataset contains of 3000 excerpts from literature, from several time periods and a wide range of reading ease scores.
The goal is to create a model that can analyze the text and predict the reading ease score of the excerpts.

For example, the excerpts "More people came to the bus stop just before 9am. Half an hour later they are all still waiting. Sam is worried." is considered to be easy to understand, so it will get high score of 1.5. On the contrary, the following excerpts: "Charge is the fundamental property of forms of matter that exhibit electrostatic attraction or repulsion in the presence of other matter." will get a negative ease score of -3.5.

The following Histogram shows the distribution of the scores:

<img width="326" alt="Capture" src="https://user-images.githubusercontent.com/71300410/122349115-97728780-cf54-11eb-934c-33277df1e046.PNG">


**Data Preprocessing**

The BERT model expects tokenized inputs with special characters ([CLS] and [SEP]), it also expect to get a padding mask that will be used in the attention module. I will use the huggingface transformers library, that make it very easy to tokenize the input sentences. It automatically adds the special character, return to tokenize sequence and the corresponding attention masks. 

**Briefly explained the Bert Model**

I will load the "bert-base-uncased" pretrained model - it is a basic English BERT model that doesn't differ between lower and upper case.

BERT is based on the original transformer paper, but only uses the encoder module. 

BERT is pre-trained on two different, but related, NLP tasks: Masked Language Modeling and Next Sentence Prediction.

The objective of Masked Language Model training is to hide a word in a sentence and then have the program predict what word has been hidden (masked) based on the hidden word's context. The objective of Next Sentence Prediction training is to have the program predict whether two given sentences have a logical, sequential connection or whether their relationship is simply random. 

Another very important detail behind BERT - it is designed to read in both directions at once. This capability is known as bidirectionality. This is where the BERT name came from - Bidirectional Encoder Representations from Transformers.

This way of training allows the model to understand context, the connection between words, and basically understand the language. 


**Loss Function**

This is a prediction of continues value task, so the bert model will have 1 output, for that we will define an MSE loss function.

**Results**

Important point - I originally trained the model with ADAM optimizer, however I couldn't get good results and it seems there wasn't obvious improvment between the epochs. Finding the ideal learning rate parameter was too difficult, so what help a lot is to use scheduler - It changed the learning rate each step according to the following pattern:


<img width="350" alt="Capture" src="https://user-images.githubusercontent.com/71300410/122358424-4c10a700-cf5d-11eb-9028-7597b034cee7.png">

The results greatly improved after using it.

The loss per epoch plot:

![loss per epoch](https://user-images.githubusercontent.com/71300410/122358998-d8bb6500-cf5d-11eb-9d7b-240ce8e33b8b.png)


I used 5 fold cross validation. The reults I got for each fold:
[0.5369,0.5507,0.4931,0.5156, 0.5441], makes the **final score 0.5281**





