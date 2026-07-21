# 100-Days-of-AI-Engineering.
I will be documenting this journey with hopefully daily updates.    

  
*Day 1 - basic linear regression model.  
It basically takes the data and predict score based on the hours inputted.  
Weak point :  
  To the model it is just some number, so if asked negative numbers of hours it will predict a negative score.  



*Day 2 - Linear regression model with two variable and error margin.  
Its quite same model that predicts the score but now based on two variable with error margin.  
Weak point :  
  Same as the first one if the the inputted values are in negative, it will heavily affect the score predicted by the model.  


*Day 3 - My first ever Torch/tensor model.  
Its functuion is nothing just it just take random data and predicts.  
Weak point :  
  Its predicts are just random because of random data inputted,  
i just wanted to create a torch model, hence this is what i spend my day on.  


*Day 4 - Just trying to get the hang of new fuctions.  
The codes main purposes is to utilize the "nn.Module" function properly.  
  Weak point :  
    It technically dont have any weak points its just unfinished.  


*Day 5 - Updated the Day 4 code.  
Now the code predicts the data 200 times to minimize the error. Whne the user inputs the hours it now shows the pridicted score.  
Weak point :  
  Still the same problem, if the user inputs a negative value it will prideict and give negative value.  


*Day 6 - Created a New model with Two hidden layers.  
Advancing into multi-layers model.  
Weak point :  
  It doesn't take any data at this point so the output will be random.  


*Day 7 - Added some more layers and functiond to the code of Day 6.  
Model trains in a loop for 500 times and get better and better. Now the predicted score also depend on another variable coffee's consumed.  
Weak point :  
  Same as the others, if provided with wrong input data the model doesnt predicts the score within 1 to 100.  

  
*Day 8 - Updated the Day 7's code to user interactive.  
Now the model trains in the loop for 500 times and the user can set both the variable ( hours study and coffe consumed ).  
I have also added ValueError so now the weak point is fixed, it doesnt allow to enter negative or unrealistic values anymore.  
Weak point :  
  Doesnt have any.   


*Day 9 - Real estate price predictor model.  
The model take the data from the CSV file called house.csv ( in the github repositorie, its call data_day9 ) loop trains and predicts the price.  
Weak point :   
  It doesnt have any weak point but it have room for improvement for user input interaction.    



(This had taken more time then i anticipated, i usally upload the coded py file at 2:00 pm but this took an entire DAY, Its 12:28 am of the next day. )  
*Day 10 - Model identifying human hand written numbers.  
So the model cant see in 2D so all the image pixle was converted in 1D array, then the model trains on MNIST files data and applise that to predict the number.  
Weak point :  
  Its too late so it is substantially finish, it currently only checks the number 7. Hopefully in the morning i would add some users interaction in the code.  


*Day 11 - Updated the Day 10 code.  
After constantly sitting in the same place for like nearly 3-4 hours the code is updated.  
Now the code asks the user for the image number ( 1 to 9999 ) then prints the image in the terminal with the actual number and then shows the AI's predicted number.  
Weak point :  
  It doesnt have any. If the image number is typed by the user incorrectly it will raise a ValueError and asks again. To exit you simply type 'quit' in the terminal and the loop breaks.   


( Note : i just remaned all the .py files with the readme and the day9 csv file too because the github was sorting them incorrectly )  


*Day 12 - Model recommending with custom embedding layers.  
Model recommends base on the data provided and trains in loops for 200 times to minimize the error then predicts what it thinks is best.  
weak point :  
  If a completely new user enters the platform with zero historical ratings, the model cannot generate an accuratly predict.  


  *Day 13 - Model predicts the nature of the sentence.  
  Model trains on specific word pattern that are coverted into mathematical tensor, and feed it into a simple neural network.  
  Weak point :  
    It doesnt have weak points but there still is room for improvement like add more words into its vocabulary.  


  *Day 14 - Bigram Language Model.  
  Model looks at a word and tries to predict very next word.  
  Weak point :  
    I have not created a larger data base in this model so it is kinda limiting.  


  *Day 15 - Model Predicts Humans nexts number.  
  Model trains on the user input and adapts in real time, then predicts.  
  Weak point :  
    It's predicts kinda useless because it predicts before the user input so the user can always change the input.  


  *Day 16 - Multi-Class Softmax Routing baseed on human sentences.  
  Model trains on the human sentences and calculate three possibility to predict.  
  Weak point :  
    It doesnt have any weak point but its very limiting.  


  *Day 17 - Cosine Similarity model.  
The model calculates the mathematical relationship between words by comparing their pre-defined vector coordinates, regardless of how they are spelled.  
Weak point :  
  It is limited by a static dictionary, if the user enter words that are not already coded in the model then it treats them like unrelated terms.  


*Day 18 - Recurrent Neural Networks.  
The model reads a sequence of words and decide if a message is Spam (1) or a normal message/Ham (0).  
Weak point :  
  Its is great at short sentences but it lacks the processing ability for longer sentences.   


Day 19 - The Query, Key, Value Engine.  
The model instead of reading a sentence one word at a time, it looks at the entire sentence at once and mathematically compares every word to every other word, allowing it to understand context by assigning importance weights to those values.  
Weak point :  
  It uses a tiny, dictionary of four words and does not actually "learn" from data, so it cannot understand any sentences outside of its pre-defined.  


*Day 20 - Transformer block.  
The model uses Residual Connections and Layer Normalization to keep the math stable, allowing data to flow through safely without degrading.  
Weak point :  
  This code has no concept of word order, so there can be misunderstanding quite often.  


*Day 21 - The positional encoder.  
It solves the previous day code problem by give all word a value for its position in the sentence.  
Weak point :   
  It doesnt have any, but it can improve a lot as there is heavily maths used so the code is not easily understandable.
