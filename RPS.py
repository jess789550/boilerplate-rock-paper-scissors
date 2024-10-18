# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# import random

# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

# def player(prev_play, opponent_history=[]):
#     opponent_history.append(prev_play)

#     # guess = "R"
#     # if len(opponent_history) > 2:
#     #     guess = opponent_history[-2]

#     prediction = random.choice(['R', 'P', 'S'])

#     if len(opponent_history) >= 1 :
#         if opponent_history[-1] == 'R':
#             prediction = random.choice(['P', 'S'])
#         elif opponent_history[-1] == 'S':
#             prediction = random.choice(['R', 'P'])
#         elif opponent_history[-1] == 'P':
#             prediction = random.choice(['R', 'S'])        
    
#     ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
#     guess = ideal_response[prediction]

#     return guess


# # taken from abbey
# def player(prev_opponent_play,
#           opponent_history=[],
#           play_order=[{
#               "RR": 0,
#               "RP": 0,
#               "RS": 0,
#               "PR": 0,
#               "PP": 0,
#               "PS": 0,
#               "SR": 0,
#               "SP": 0,
#               "SS": 0,
#           }]):

#     if not prev_opponent_play:
#         prev_opponent_play = 'R'
#     opponent_history.append(prev_opponent_play)

#     last_two = "".join(opponent_history[-2:])
#     if len(last_two) == 2:
#         play_order[0][last_two] += 1

#     potential_plays = [
#         prev_opponent_play + "R",
#         prev_opponent_play + "P",
#         prev_opponent_play + "S",
#     ]

#     sub_order = {
#         k: play_order[0][k]
#         for k in potential_plays if k in play_order[0]
#     }

#     prediction = max(sub_order, key=sub_order.get)[-1:]

#     ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
#     return ideal_response[prediction]


# DeepAI modification of abbey
# import random
# from collections import Counter

# def player(prev_opponent_play, opponent_history=None):
#     if opponent_history is None:
#         opponent_history = []

#     if prev_opponent_play == "":
#         prev_opponent_play = 'R'  # Default play

#     opponent_history.append(prev_opponent_play)

#     # Define ideal responses
#     ideal_response = {'R': 'P', 'P': 'S', 'S': 'R'}

#     # Analyze history for the opponent's play frequency
#     if len(opponent_history) < 2:
#         return ideal_response[prev_opponent_play]  # Respond based on the last move if less than 2 rounds

#     # Count the frequency of each move
#     frequency = Counter(opponent_history)
    
#     # Predict the opponent's next move based on the most frequent last move
#     most_common_move = frequency.most_common(1)[0][0] if frequency else 'R'

#     # Determine the ideal response
#     prediction = ideal_response[most_common_move]

#     # Introducing randomness to the response for increased unpredictability
#     if random.random() < 0.1:  # 10% chance to play a random move
#         return random.choice(['R', 'P', 'S'])

#     return prediction




# DeepAI
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# import random

# class RPSPlayer:
#     def __init__(self):
#         self.opponent_history = []
#         self.moves = ['R', 'P', 'S']
#         self.model = RandomForestClassifier()

#     def update_history(self, prev_play):
#         # Update opponent history
#         self.opponent_history.append(prev_play)

#     def train_model(self):
#         # Prepare data for training the model
#         if len(self.opponent_history) < 2:
#             return  # Not enough data to train the model

#         # Encode the moves to numeric values
#         label_encoder = LabelEncoder()
#         encoded_moves = label_encoder.fit_transform(self.opponent_history)

#         # Define features and target
#         X = []
#         y = []
        
#         for i in range(1, len(encoded_moves)):
#             X.append([encoded_moves[i-1]])  # Previous move
#             y.append(encoded_moves[i])       # Current move

#         X = np.array(X)
#         y = np.array(y)

#         # Train the model
#         self.model.fit(X, y)

#     def predict_next_move(self):
#         if len(self.opponent_history) < 1:
#             return random.choice(self.moves)  # Return a random choice if no history

#         # Predict the next move based on the last move
#         last_move_encoded = np.array([[self.opponent_history[-1]]])  # Last opponent move
#         predicted_move_encoded = self.model.predict(last_move_encoded)

#         # Decode the predicted move back to original
#         return self.moves[predicted_move_encoded[0]]

#     def counter_move(self, opponent_move):
#         if opponent_move == 'R':
#             return 'P'  # Paper beats Rock
#         elif opponent_move == 'P':
#             return 'S'  # Scissors beat Paper
#         elif opponent_move == 'S':
#             return 'R'  # Rock beats Scissors

#     def player(self, prev_play):
#         self.update_history(prev_play)
#         self.train_model()
#         predicted_move = self.predict_next_move()
#         return self.counter_move(predicted_move)

# # Example of how you might use the class
# player = RPSPlayer()
# for i in range(10):  # Simulate 10 rounds
#     # Simulate a random opponent's previous move for demo
#     opponent_move = random.choice(['R', 'P', 'S'])
#     print(f"Opponent's move: {opponent_move}")
#     move = player_instance.player(opponent_move)
#     print(f"Player's counter move: {move}")



# https://medium.com/@sri.hartini/rock-paper-scissors-in-python-5173ab69ca7a
def player(prev_play, opponent_history=[], play_order={}):
    # first round set rock as default move by opposition
    if not prev_play:
        prev_play = 'R'
    
    # set paper as default move by opposition for 4 rounds
    opponent_history.append(prev_play)
    prediction = 'P'
    
    # analyse opponent's last 5 moves and find most frequent moves
    if len(opponent_history) > 4:
        last_five = "".join(opponent_history[-5:])
        play_order[last_five] = play_order.get(last_five, 0) + 1
        
        potential_plays = [
            "".join([*opponent_history[-4:], v]) 
            for v in ['R', 'P', 'S']
        ]

        sub_order = {
            k: play_order[k]
            for k in potential_plays if k in play_order
        }

        if sub_order:
            prediction = max(sub_order, key=sub_order.get)[-1:]
    
    # if the opponent is predicted to play Paper then play Scissors etc.
    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}

    return ideal_response[prediction]
