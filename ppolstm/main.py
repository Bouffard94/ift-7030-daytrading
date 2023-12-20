from model import PPO
from data.dailytickerdataset import DailyTickerDataset
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np
from math import floor
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# params
input_dims = 9
hidden_size = 32
lstm_num_layers= 2
alpha = 1e-3
gamma = 0.99
lamb = 0.95
policy_clip = 0.2
batch_size = 1
n_epochs = 1
N = 6 * 60 # 1h = 10s * 6 * 60min
actions = [0.20, 0.10, 0.05, 0, -0.25, -0.5, -1]

ppo = PPO(len(actions), input_dims, hidden_size, lstm_num_layers, gamma, lamb, alpha, policy_clip, batch_size)

print("Building dataset...")
csv_file = 'ppolstm/data/csv/common_10s_202312192103.csv'
dataset = DailyTickerDataset.from_csv(csv_file, nrows=100000)

# train_dataset, test_dataset = random_split(dataset, [floor(len(dataset) * 0.75), floor(len(dataset) * 0.25)])
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

train_loader = DataLoader(dataset, batch_size=batch_size)

# print(f"Train: Nb ticker/day combinations: {len(train_dataset)}")
# print(f"Test: Nb ticker/day combinations: {len(test_dataset)}")

print("Training...")

profit_history = []
episode = 0
for day_data in train_loader:
    episode += 1
    #print(f"Episode: {episode}/{len(train_loader)}")
    day_data: torch.Tensor = day_data.squeeze().to(device)

    # Init porfolio
    balance = 10000.
    n_positions = 0
    price = 1.
    best_profit = 0
    worst_profit = 0
    total_profit = 0
    porfolio_value = balance + price * n_positions

    starting_porfolio_value = porfolio_value
    balance = torch.tensor(balance).to(device)
    learn_iters = 0
    for t in range(day_data.shape[0]):
        # Adjust price
        closing_price_delta = day_data[t, 3]
        price *= (1 + closing_price_delta)

        # Calculate new portfolio value and profit
        holdings = price * n_positions
        new_porfolio_value = balance + holdings
        profit = new_porfolio_value / porfolio_value - 1
        porfolio_value = new_porfolio_value
        total_profit += profit.item()

    
        # Remember state, action and profit
        done = t == day_data.shape[0] - 1
        if t != 0:
            ppo.remember(state, action, probs, value, profit, done)

        # Learn at each N time step
        if t % N == 0 and t != 0:
            #print(f"Learning: {(t - N)/6}min to {t/6} min")
            ppo.learn(n_epochs)
            learn_iters += 1

        # Set state
        state = torch.cat([balance.unsqueeze(0), holdings.unsqueeze(0), day_data[t]]).to(device)

        # Take action (buy/sell n stock)
        action, probs, value = ppo.select_action(state)
        r_action = actions[action]
        if r_action > 0: # we buy with r% of balance
            purchase_power = balance * r_action
            buy = floor(purchase_power / price)
            balance -= buy * price
            n_positions += buy
        elif r_action < 0: # we sell -r% of positions 
            sell = -floor(n_positions * r_action)
            balance += sell * price
            n_positions -= sell

    if total_profit > best_profit:
        best_profit = total_profit
    if total_profit < worst_profit:
        worst_profit = total_profit
    #     ppo.save_models()

    porfolio_val = round(porfolio_value.item(), 2)
    profit = round(porfolio_value.item() - starting_porfolio_value, 2)
    profit_perc = round(total_profit * 100, 3)
    print(f"Fin episode {episode}\t Total:{porfolio_val}$\t Profit: {profit}$, {profit_perc}%")

    profit_history.append(profit)

print("Done training!")
#ppo.save_models()

indices = np.arange(len(profit_history))
above_zero_profits = [i for i, value in enumerate(profit_history) if value >= 0]
below_zero_profits = [i for i, value in enumerate(profit_history) if value < 0]
plt.fill_between(indices, profit_history, where=[val >= 0 for val in profit_history], color='green', label='Above 0', interpolate=True)
plt.fill_between(indices, profit_history, where=[val < 0 for val in profit_history], color='red', label='Below 0', interpolate=True)
plt.axhline(y=0, color='gray', linestyle='--', label='0')
plt.xlabel('épisodes')
plt.ylabel('profit ($)')
plt.title('Profit par épisodes')
plt.show()
