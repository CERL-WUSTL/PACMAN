import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure the embeddings directory exists
os.makedirs('embeddings', exist_ok=True)

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

# Set the model to evaluation mode
model.eval()

# Define the tasks with their objectives and details
tasks = {
    "Reach": "Objective: Move the robot's end-effector to a target position. Environment Details: The task is set on a flat surface with no obstacles. The target position is marked by a small sphere or point in space. The robot must move its arm to touch this point.",
    "Push": "Objective: Push an object to a specified goal position on the table. Environment Details: The object starts in a fixed position on a flat table. The goal position is marked on the table's surface. The robot must apply force to the object to move it to the goal without lifting it.",
    "Pick-Place": "Objective: Pick up an object and place it at a designated goal position. Environment Details: The object is placed randomly on the table. The goal position is marked by a target area. The robot must grasp the object using its gripper, lift it, and then place it accurately in the target area.",
    "Door Open": "Objective: Open a door by pulling or pushing it. Environment Details: The door is hinged and can be pulled or pushed open. The robot must grasp the handle and apply force in the correct direction to open the door.",
    "Drawer Open": "Objective: Open a drawer by pulling it. Environment Details: The drawer is initially closed and can slide out on rails. The robot must grasp the handle and pull it outward to open the drawer.",
    "Drawer Close": "Objective: Close an open drawer by pushing it. Environment Details: The drawer starts in an open position. The robot must apply force to the handle or the front of the drawer to push it back into the closed position.",
    "Button Press": "Objective: Press a button by moving the end-effector. Environment Details: The button is mounted on a panel or flat surface. The robot must position its end-effector above the button and apply downward force to press it.",
    "Peg Insertion Side": "Objective: Insert a peg into a hole from the side. Environment Details: The peg and hole are aligned horizontally. The robot must grasp the peg and insert it into the hole by approaching from the side.",
    "Window Open": "Objective: Slide a window open. Environment Details: The window is set within a frame and can slide horizontally. The robot must apply lateral force to slide the window open.",
    "Window Close": "Objective: Slide a window closed. Environment Details: The window starts in an open position and can slide horizontally within its frame. The robot must apply lateral force to slide the window shut."
}

# Function to get embeddings for a given text
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Get the hidden states of the last layer
    hidden_states = outputs.hidden_states[-1]
    return torch.mean(hidden_states, dim=1)

# Iterate over tasks and save their embeddings
for task_name, task_text in tasks.items():
    embeddings = get_embeddings(task_text)
    torch.save(embeddings, f'embeddings/{task_name}.pth')

print("Embeddings saved successfully.")
