
import pygame
import random
import sys
import pickle
# parameters
STATE_SIZE = 4
ACTION_SIZE = 3
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.99
BATCH_SIZE = 32
MEMORY_SIZE = 10000
BALL_SPEED = 10
TRAIN_UNTIL_SCORE = 40
RND_DIR_LIST = (-1,-0.8,-0.6,0.6,0.8, 1)

# colors
colors = {
    "red": (255, 0, 0),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "black": (0, 0, 0)
}


# main game class
class BrickBreaker:
    def __init__(self, width, height,ai_model=None):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Brick Breaker")
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.score = 0
        self.game_over = False
        self.ball = pygame.Rect(self.width/2-10, self.height-50, 20, 20)
        self.rnd_dir = random.choice(RND_DIR_LIST)
        self.ball_speed_x = BALL_SPEED *  self.rnd_dir
        self.ball_speed_y = -BALL_SPEED
        self.paddle = pygame.Rect(self.width/2-70, self.height-20, 140, 10)
        self.brick_list = []
        self.create_bricks()

        self.ai = BrickBreakerAI(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, DISCOUNT_RATE, BATCH_SIZE, MEMORY_SIZE)
        if ai_model is not None:
            self.ai.model = ai_model
            self.ai.target_model = ai_model
        self.state = self.get_state()
        self.action = 0
        self.reward = 0
        self.next_state = self.get_state()
        self.done = False
        


    def create_bricks(self):
            for row in range(7):
                if row % 3 != 0:
                    for col in range(17):
                        if col % 4 != 0:
                            brick = pygame.Rect(20 + col * (40 + 5), 50 + row * (20 + 5), 40, 20)
                            self.brick_list.append(brick)

    def draw_bricks(self):
        for brick in self.brick_list:
            pygame.draw.rect(self.screen, colors["red"], brick)

    def draw_paddle(self):
        pygame.draw.rect(self.screen, colors["yellow"], self.paddle)


    def draw_ball(self):
        # with radius 10
        pygame.draw.circle(self.screen, colors["green"], (self.ball.x + 10, self.ball.y + 10), 10)

   

    def move_ball(self):
        self.ball.x += self.ball_speed_x * abs(self.rnd_dir)
        index = self.ball.collidelist(self.brick_list)
        self.ball.y += self.ball_speed_y
        if index != -1:
            self.score += 3
            self.brick_list.pop(index)
            self.ball_speed_x *= -1

        else:
            index = self.ball.collidelist(self.brick_list)
            if index != -1:
                self.score += 3
                self.brick_list.pop(index)
                self.ball_speed_y *= -1

        if self.ball.x <= 0:
            self.ball.x = 0
            self.ball_speed_x *= -1
        elif self.ball.x >= self.width - 20:
            self.ball.x = self.width - 20
            self.ball_speed_x *= -1
        if self.ball.y <= 0:
            self.ball.y = 0
            self.ball_speed_y *= -1
        elif self.ball.y >= self.height - 20:
            self.ball.y = self.height - 20
            self.ball_speed_y *= -1
            self.game_over = True
        if self.ball.colliderect(self.paddle):
            self.ball.y = self.height - 50
            self.reward += 3
            self.ball_speed_y *= -1


    def reset(self):
        self.score = 0
        self.game_over = False
        self.ball.x = self.width/2-10
        self.ball.y = self.height-50
        self.rnd_dir = random.choice(RND_DIR_LIST)
        self.ball_speed_x = BALL_SPEED *  self.rnd_dir
        self.ball_speed_y = -BALL_SPEED
        self.paddle.x = self.width/2-70
        self.brick_list = []
        self.create_bricks()

    def get_state(self):
        state = np.array([self.paddle.x, self.ball.x, self.ball.y, self.ball_speed_x])
        state = state.reshape(1, STATE_SIZE)
        return state
    
    def step(self, action):
        self.action = action
        if self.action == 0:
            self.paddle.x -= 10
        elif self.action == 1:
            self.paddle.x += 10
    
        if self.paddle.x <= 0:
            self.paddle.x = 0
        elif self.paddle.x >= self.width - 140:
            self.paddle.x = self.width - 140
        
    # ai game run
    def run(self):
        tempArray = []
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.screen.fill(colors["black"])
            self.draw_bricks()
            self.draw_paddle()
            self.draw_ball()
            self.move_ball()
            self.state = self.next_state
            self.next_state = self.get_state()
            self.done = False

            if len(self.brick_list) == 0:
                self.reset()
                self.done = True
                self.reward = 100
            if self.game_over:
                tempArray.append(self.score)
                self.reward = -100
                self.ai.remember(self.state, self.action, self.reward, self.next_state, self.done)
                self.ai.replay(BATCH_SIZE)
                self.reward = 0
                self.reset()
                self.done = True
                
            # draw score
            font = pygame.font.SysFont("Arial", 30)
            score_label = font.render(f"Score: {self.score} Reward:{self.reward}", 1, colors["orange"])
            self.screen.blit(score_label, (10, 10))
            self.step(self.ai.act(self.state))
            pygame.display.update()
            self.clock.tick(60)

            print(self.score)
            if self.score >= 100000:
                # Save the model
                with open("model2.pkl", "wb") as file:
                    pickle.dump(self.ai.model, file)
                print(tempArray)
                pygame.quit()
                sys.exit()
                        

        
# ai class
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random


class BrickBreakerAI:
    def __init__(self, state_size, action_size, learning_rate, discount_rate, batch_size, memory_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def act(self, state):
        return self.choose_action(state, epsilon=0.1)

    def replay(self, batch_size):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        targets = rewards + self.discount_rate * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
# main function
import os 
def main():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as file:
            ai_model = pickle.load(file)
        game = BrickBreaker(800, 600, ai_model = ai_model)
    else:
        game = BrickBreaker(800, 600)
    game.run()



if __name__ == "__main__":
    main()