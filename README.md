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
Weak point : Still the same problem, if the user inputs a negative value it will prideict and give negative value.  


*Day 6 - Created a New model with Two hidden layers.  
Advancing into multi-layers model.  
Weak point : It doesn't take any data at this point so the output will be random.  


*Day 7 - Added some more layers and functiond to the code of Day 6.  
Model trains in a loop for 500 times and get better and better. Now the predicted score also depend on another variable coffee's consumed.  
Weak point : Same as the others, if provided with wrong input data the model doesnt predicts the score within 1 to 100.  

  
*Day 8 - Updated the Day 7's code to user interactive.  
Now the model trains in the loop for 500 times and the user can set both the variable ( hours study and coffe consumed ).  
I have also added ValueError so now the weak point is fixed, it doesnt allow to enter negative or unrealistic values anymore.  
Weak point : Doesnt have any.   


*Day 9 - Real estate price predictor model.  
The model take the data from the CSV file called house.csv ( in the github repositorie, its call data_day9 ) loop trains and predicts the price.  
Weak point : It doesnt have any weak point but it have room for improvement for user input interaction.
