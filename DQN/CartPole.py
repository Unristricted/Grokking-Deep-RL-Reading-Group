import gym
import pygame
import numpy as np
import time
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize the gym environment for CartPole
env = gym.make('CartPole-v1')

# Initialize Pygame
pygame.init()

# Set the screen size and initialize the display
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("CartPole Game")

# Game loop
done = False

# Reset the environment to start with a fresh state
state, _ = env.reset()

# Initialize the clock for smooth frame rate
clock = pygame.time.Clock()

# Timer and score variables
start_time = time.time()
total_time = 0
score = 0

# Keyboard controls: left arrow (move left) and right arrow (move right)
action_map = {pygame.K_LEFT: 0, pygame.K_RIGHT: 1}

font = pygame.font.SysFont("Arial", 24)

# Constants to slow down the pole's movement
MAX_POLE_ANGLE = 0.15  # Limit the pole angle for smoother motion
MAX_CART_VELOCITY = 2.0  # Limit the cart's velocity to prevent it from moving too fast

# Time step for simulation to slow down the updates
TIME_STEP = 0.02  # Lower time step for a smoother update

# Game loop with continuous environment reset when the game is done
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Get keyboard input for controlling the cart
    action = None
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action = 0  # Move cart left
    elif keys[pygame.K_RIGHT]:
        action = 1  # Move cart right

    # Perform the action
    next_state, reward, done, _, _ = env.step(action)

    # Update the state and score
    state = next_state
    score += reward

    # Calculate elapsed time
    total_time = time.time() - start_time
    total_time = round(total_time, 1)

    # Render the environment to the screen
    screen.fill((255, 255, 255))  # Fill the screen with white

    # Extract the state information
    cart_position, cart_velocity, pole_angle, pole_velocity = state

    # Limit the pole angle and cart velocity for smoother movement
    pole_angle = np.clip(pole_angle, -MAX_POLE_ANGLE, MAX_POLE_ANGLE)
    cart_velocity = np.clip(cart_velocity, -MAX_CART_VELOCITY, MAX_CART_VELOCITY)

    # Draw the cart
    cart_x = int(SCREEN_WIDTH / 2 + cart_position * 50)
    cart_y = int(SCREEN_HEIGHT - 50)

    pygame.draw.rect(screen, (0, 0, 0), (cart_x - 40, cart_y - 20, 80, 20))

    # Draw the pole
    pole_length = 100
    pole_x = cart_x
    pole_y = cart_y - 20
    pole_end_x = pole_x + pole_length * np.sin(pole_angle)
    pole_end_y = pole_y - pole_length * np.cos(pole_angle)

    pygame.draw.line(screen, (255, 0, 0), (pole_x, pole_y), (pole_end_x, pole_end_y), 6)

    # Draw the timer and score
    timer_text = font.render(f"Time: {total_time}s", True, (0, 0, 0))
    screen.blit(timer_text, (10, 10))

    score_text = font.render(f"Score: {int(score)}", True, (0, 0, 0))
    screen.blit(score_text, (SCREEN_WIDTH - 150, 10))

    # Update the display with double buffering for smoother rendering
    pygame.display.update()

    # Slow down the game by adjusting the frame rate
    clock.tick(60)

    # If the game is done, reset it
    if done:
        state, _ = env.reset()
        start_time = time.time()
        total_time = 0
        score = 0
        done = False

# Show the final statistics (time and score) when the game ends
final_time = round(total_time, 1)
final_score = int(score)

# Display final stats
screen.fill((255, 255, 255))  # Fill the screen with white for the final screen
final_text = font.render(f"Game Over! Final Time: {final_time}s", True, (0, 0, 0))
screen.blit(final_text, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 20))

final_score_text = font.render(f"Final Score: {final_score}", True, (0, 0, 0))
screen.blit(final_score_text, (SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT // 2 + 20))

pygame.display.update()

# Wait for a few seconds before closing the game
pygame.time.delay(3000)

env.close()
pygame.quit()