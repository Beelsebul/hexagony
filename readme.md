# Hexagony
> Exploring the potential of transformer-based models to understand game strategy and improve performance through self-playing.

This project investigates Transformer-based language models for Hex, a game with a simple action space and deep strategy. Our goal is to find the best way to train these models. We compile a dataset of over 25,000 complete games and train separate GPT-style models using various labeling techniques. We then refine these models using an iterative self-play pipeline with fine-tuning on both self-generated winning games and the original dataset. Evaluated on test cases with likely and unseen openings, the refined agents outperform earlier generations and are comparable to baseline algorithms.
 


## Folders available for review:

This section provides a description of all elements in this project, including code, graphs, and models.


- Training - Contains all scripts related to model creation and game generation using already trained models.
- testing_and_evaluating - Contains all scripts for data processing, evaluation, and model testing.
- Logs, Graphs - Contain textual and visual representations of model testing as well as some loss graphs from training.
- Website - Contains files for initializing a local server as well as a page template for playing the board game.
- mini_red_test and mini_blue_test - Contain checkpoints of different generations.



Files in the Training folder:

1. tokenizer.py - Used for training and creating a tokenizer, saving it in the tokenizer folder.

2. train_model.py - Responsible for training an initial model based on starting games won by the red player, with random initial weight initialization. Saves the model to the minihex_red folder and training data in logs.

3. generate_new_games.py - Responsible for generating new games between two models by selecting more than one most likely move for the red model. Generated games are saved to a CSV file in logs.

4. fine_tune.py - Responsible for fine-tuning the model based on new games won by the red player, loading weights from the previous model. Saves the model to the minihex_red folder and training data in logs.

   
Files in the Testing_and_Evaluating folder:

1. test_model.py - The first model testing implementation, which provides the top 5 predictions for a game entered in the input field.

2. test_tokenizer.py - Created to view and verify the tokenization process.

3. test_agents.py - Testing a model against a deep reinforcement learning model or against the original algorithm

4. checkpoint_tournament.py - Similar to testwinner.py, but optimized by ChatGPT and works several times faster.

5. heatmap.py - Displays the same results matrix but represents relationships with colors for easier data visualization.

6. elo_compute.py - Calculates model rankings using the Elo rating based on the matrix obtained from testwinner, simplifying model comparison from matrix form to numeric representation.

7. attention_heads.py - Gives a graphical representation of attention heads for any game sequence

   
Files in the Website folder:

1. app.py - Starts the webserver which then enables user to test the model via visual interface.
  
2. static and templates - Contain files required to build visual interface including the game logics and board Generation (the board is resizable)
