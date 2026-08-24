_# 100-Days-of-AI-Engineering.
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
(After constantly sitting in the same place for like nearly 3-4 hours the code is updated.)  
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


*Day 22 - Next word predictor.  
The model takes a row of words converts them into digital coordinates then adds a time-stamp to locate words order. In the end, it predicts which word makes the most sense to come next in a sentence.  
Weak point :  
  It is not fully finished, it needs a blinder for the next words as now the model unintentionally cheats by looking at the next word.  


*Day 23 - Blinder.  
It doesnt let the model cheat as it was doing in the previous code.  
Weak point :  
  It misunderstands sometimes if the sentence meaning is at the end.  


*Day 24 - Training LLM.  
It trains on a sentence and generates the sentence on what promt was given.  
Weak point :  
  As there is quite literally nothing to train on except of one sentence, if the promt is a different word then it malfunctions.  


*Day 25 - Temperature and Sampling.  
I just wanted to learn about temp and sampling,so yeah i added it to the code. In short it is lottery, as the temp increase the odds decrease.  
Weak point :  
  As i said earlier, if the temp is too high then the picked word will be fully random.  


*Day 26 - Top-K and Top-P.  
Top-K filters outs the top words from rest and Top-P combines the top words confidence to hit the targeted percentage.  
Weak point :  
  If the Top-K value is too low then the model will become a repetitive program.  


*Day 27 - Fine-Tuner.  
It is just a chatbot which trains on pre-questions and their answers, to provede assistance.  
Weak point :  
  As i said earlier, the model needs the data before asking question to it. You would have to update it's data and train it everytime if you want to add sometihing new.  
  Plus it is still partially incompelet, it stil need to answer the question that it currently just understands.  


*Day 28 - Retrieval-Augmented Generation.  
I wanted to try RAG for my LLM as the RAG make the AI see in the databse for answers, it can be very sensitive as the model's response solely depend on the database.  
Weak point :  
  The model is not working as i expected, there are data provided but it doesnt recognises it. Hopewfully i can fix it in tomorrow's code.  


*Day 29 - Updated the Day 28's code.  
I fixed the "no match" and also added user input in the code, now the the user can type what they want to know ( even though its very limiting )  
Weeak point :  
  The code is just a little updated not changed so it has the same weak point as the Day 28 code.  


*Day 30 - React Framework.  
Created an AI agent that can use calculator if it is required or needed and returns the answers.  
Weak point :  
  It can get in infinite looping as the code tells the AI to use calculator if needed ( if the question is a mathematical question then every time ) and the calculator tool is not working then it gets stuck in a loop.  


*Day 31 - Vision Transformers.  
Same as I did in the Day 10 and 11, it convert the 2D image into 1D line to understand the images.  
Weak point :  
  It doesnt have any, but if an object is far away and is also split into two pixel then the model takes a lot longer to identify that object.  


*Day 32 - The Multimodal Bridge.  
I learned this from Youtube and this pretty simple, it breaks the image as the previous day code and also limits the search on the image according to the questions asked.  
Weak point :  
  The resolution of the image can affect the time the processing takes to answer the question.  


*Day 33 - The Diffusion Engine.  
The diffusion engine adds static noise and the model have to reverse the process by removing the noise ( the model doesnt know what the image looks like. )  
Weak point :  
  The image quality can be affected by the complexity of the images and can also take more time to processs.  


*Day 34 - The Conditioning Engine.  
Its the guide to how to generate an image according to its promt.  
Weak point :  
  It is limited by its vocabulary, if the promt is something different from the database then it totally ignores or generate something random.  


*Day 35 - The LoAR.  
To teach old AI new things it needs to be retrained, but LoAR creates a small model and then trains it on the new things and adds it to the old AI.  
Weak point :  
  When LoAR creates a small side model to teach, it can only teach small tasks as its only a side model.  


*Day 36 - Spectrogram.  
So this code make the audio mapped out on an imagen, making it easier for AI to train.  
Weak point :  
  It only prints the high pitches, low pitches, and volume over time. It doesnt print phase ( Phase tells you the exact microscopic timing of the sound waves )  


*Day 37 - Compressing.  
I wanted to make this model long ago, but anyway it cuts the decimals into whole simple number. It does reduces the pression but it allows to reduce the size of the model, so it can be shared and run easily.  
Weak point :  
  As I said earlier, compressing reduces pression and if compressed to much then it can break the model completely.  


*Day 38 - Key-Value Caching.  
It creates a saved cache in the memory of its previous work, if continuing from the same work it it doesnt restart and uses the saved cache to resume. Therefore I wanted to try this as its quite fast.  
Weak point :  
  It saves the work in the memory and if the work is too massive then it will affect the device as well as might even crash it if not careful.  


*Day 39 - Window mask.  
This model only looks at the small chunk of recent text ( 500/10,000 ), preventing the computer from running of memory while processing.  
Weak point :  
  The model cant look back, so if there was info on the previous page and ask the model about it in the next page it will hallucinate or fail to answer.  


*Day 40 - The Expert Router.  
The model splits the brain into smaller and specialized section. When the user asks a question, the router selects the 2 specific section that are best suited.  
Weak point :  
  Even if the model's 75% brain is asleep it does take 100% of the files into memory, so it doesnt save any RAM.  


*Day 41 - The Judge.  
This model tracks another model and reward or punish based on the output. If the output is toxic it punishes the model and teaches it to act polite.  
Weak point :  
  The judge is very fragile, it also requires to run two neural networks simultaneously. If the sentence is phrase to look good but it is violent then the judge misunderstands it.  


*Day 42 - Direct Preference Optimization.  
It shows the model a good answer and a bad answer side by side and the math forces the model to favor the good answer while suppressing the bad one .  
Weak point :  
  It can take a lot of memory as if the model is X amount then it would require 2X amount in memory simultaneously.  


*Day 43 - State Space Model.  
In short it doesnt increase the memory usage while feeding the model new words.  
Weak point :  
  The model wont keep words that it thinks are useless or isnt important enough to keep in the memory box.  


*Day 44 - Vector Search.  
Another search/predict model but i found this on reddit and its little different. It searches by meaning rather than by exact words, it mathematically translates paragraphs into coordinates on a massive graph to search.  
Weak point :  
  It is terrible at finding exact, specific numbers or IDs.  


*Day 45 - The Two-Tower Engine.  
This model uses two separate neural networks to turn question and documents into numbers and train it by pulling correct answers closer and pushing wrong answer far away.  
Weak point :  
  As the two neural networks are completely separately, this causes the AI to miss subtle details and complex word relationships.  

*Day 46 - Cross Encoder.  
It puts given question and the document together into one sentence and feeds it to the AI, this process lets the AI see exactly how the words in given question relate to the words in the document.  
Weak point :  
  It is incredibly slow, for example if wanting to search database of 1 million documents, a Cross-Encoder has to run 1 million complete neural network calculations from scratch.  


*Day 47 - TransE.  
It is basically instead of sorting messy paragraphs of text, it stores the exact facts as connected dots, this makes it really fast.  
Weak point :  
  As I said earlier it saves the exact words while connecting, if the words are emotional or defining a feeling then it can get slow and can also malfunction.  


*Day 48 - The Dropout Engine.  
I didnt know that Dropout feature was already built in Pytorch so I had to try it. So it basically make the model to learn and prevents just memorizing by adding randomness.  
Weak point :  
  It didnt had any weakness but this process is time consuming and takes more computing power.  


*Day 49 - Temperature and Top-P.  
Temperature increase the probability of picking the words that is not top 1, this makes less repetitive and more creative. Top-P acts as a safety net by completely removing absolute worst words.  
Weak point :  
  If the Temperature is set too high then it spits out random letters and hallucinated garbage, and if set too low then the model becomes incredibly repetitive.  


*Day 50 - Transformer.  
It generates words by turning text into math and then figuring out context. It is nothing very interesting but i wanted to try it out.  
Weak point :  
  It only generates one word at a time, so to generate a full sentence it would generate one word then redo all the process for the second word and for all the remaining words.  


*Day 51 - Multimodel.  
It just converts the image into a mathematical language of text.  
Weak point :  
  It can miss some details from the image as it is compressing the image.  


*Day 52 - Beam Search.  
It generates 3 to 5 sentences and explore them at the same time, by this the final sentence always make grammatical sense from beginning to end.  
Weak point :  
  It generates multiple sentences at the same time, making it very power hungry.  


*Day 53 - Gradient Accumulation.  
It saves memory by training a model in small mini batches.  
Weak point :  
  It can take longer than usual as it trains a small batch and clears the GPU memory and starts train.  


*Day 54 - Mixed Precision.  
To save time time the AMP ( Automatic Mixed Precision ) uses FP16 for the heavy lifting, but at the end switches back to highly precise math FP32 to save the final answer.  
Weak point :  
  If the data provided is not cleared or if the values are so small, then the computer will rounds then down to absolute zero ( Gradient Underflow ), making model stop learning and breaks it.
