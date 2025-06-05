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
    "Reach-v1": "Objective: Move the robot's end-effector to a target position. Environment Details: The task is set on a flat surface with random goal positions. The target position is marked by a small sphere or point in space.",
    "Push-v1": "Objective: Push a puck to a specified goal position. Environment Details: The puck starts in a random position on a flat surface. The goal position is marked on the surface. The robot must apply force to the puck to move it to the goal.",
    "Pick-Place-v1": "Objective: Pick up a puck and place it at a designated goal position. Environment Details: The puck is placed randomly on the surface. The goal position is marked by a target area. The robot must grasp the puck using its gripper, lift it, and place it in the target area.",
    "Door-Open-v1": "Objective: Open a door with a revolving joint. Environment Details: The door can be opened by rotating it around the joint. The robot must grasp the handle and apply force to rotate the door open. Door positions are randomized.",
    "Drawer-Open-v1": "Objective: Open a drawer by pulling it. Environment Details: The drawer is initially closed and can slide out on rails. The robot must grasp the handle and pull it outward to open the drawer. Drawer positions are randomized.",
    "Drawer-Close-v1": "Objective: Close an open drawer by pushing it. Environment Details: The drawer starts in an open position. The robot must apply force to the handle or the front of the drawer to push it back into the closed position. Drawer positions are randomized.",
    "Button-Press-Topdown-v1": "Objective: Press a button from the top. Environment Details: The button is mounted on a panel or flat surface. The robot must position its end-effector above the button and apply downward force to press it. Button positions are randomized.",
    "Peg-Insert-Side-v1": "Objective: Insert a peg into a hole from the side. Environment Details: The peg and hole are aligned horizontally. The robot must grasp the peg and insert it into the hole by approaching from the side. Peg and goal positions are randomized.",
    "Window-Open-v1": "Objective: Slide a window open. Environment Details: The window is set within a frame and can slide horizontally. The robot must apply lateral force to slide the window open. Window positions are randomized.",
    "Window-Close-v1": "Objective: Slide a window closed. Environment Details: The window starts in an open position and can slide horizontally within its frame. The robot must apply lateral force to slide the window shut. Window positions are randomized.",
    "Door-Close-v1": "Objective: Close a door with a revolving joint. Environment Details: The door can be closed by rotating it around the joint. The robot must grasp the handle and apply force to rotate the door shut. Door positions are randomized.",
    "Reach-Wall-v1": "Objective: Bypass a wall and reach a goal position. Environment Details: The goal is positioned behind a wall. The robot must maneuver around the wall to reach the goal. Goal positions are randomized.",
    "Pick-Place-Wall-v1": "Objective: Pick a puck, bypass a wall, and place the puck at a goal position. Environment Details: The puck and goal are positioned with a wall in between. The robot must navigate around the wall to complete the task. Puck and goal positions are randomized.",
    "Push-Wall-v1": "Objective: Bypass a wall and push a puck to a goal position. Environment Details: The puck and goal are positioned with a wall in between. The robot must maneuver around the wall to push the puck to the goal. Puck and goal positions are randomized.",
    "Button-Press-v1": "Objective: Press a button. Environment Details: The button is mounted on a panel or surface. The robot must position its end-effector above the button and apply force to press it. Button positions are randomized.",
    "Button-Press-Topdown-Wall-v1": "Objective: Bypass a wall and press a button from the top. Environment Details: The button is positioned behind a wall on a panel. The robot must maneuver around the wall to press the button. Button positions are randomized.",
    "Button-Press-Wall-v1": "Objective: Bypass a wall and press a button. Environment Details: The button is positioned behind a wall. The robot must navigate around the wall to press the button. Button positions are randomized.",
    "Peg-Unplug-Side-v1": "Objective: Unplug a peg sideways. Environment Details: The peg is inserted horizontally and needs to be unplugged. The robot must grasp the peg and pull it out sideways. Peg positions are randomized.",
    "Disassemble-v1": "Objective: Pick a nut out of a peg. Environment Details: The nut is attached to a peg. The robot must grasp and remove the nut. Nut positions are randomized.",
    "Hammer-v1": "Objective: Hammer a nail on the wall. Environment Details: The robot must use a hammer to drive a nail into the wall. Hammer and nail positions are randomized.",
    "Plate-Slide-v1": "Objective: Slide a plate from a cabinet. Environment Details: The plate is located within a cabinet. The robot must grasp the plate and slide it out. Plate and cabinet positions are randomized.",
    "Plate-Slide-Side-v1": "Objective: Slide a plate from a cabinet sideways. Environment Details: The plate is within a cabinet and must be removed sideways. The robot must maneuver the plate out. Plate and cabinet positions are randomized.",
    "Plate-Slide-Back-v1": "Objective: Slide a plate into a cabinet. Environment Details: The robot must place the plate back into a cabinet. Plate and cabinet positions are randomized.",
    "Plate-Slide-Back-Side-v1": "Objective: Slide a plate into a cabinet sideways. Environment Details: The plate is positioned for a sideways entry into the cabinet. The robot must maneuver the plate into place. Plate and cabinet positions are randomized.",
    "Handle-Press-v1": "Objective: Press a handle down. Environment Details: The handle is positioned above the robot's end-effector. The robot must apply downward force to press the handle. Handle positions are randomized.",
    "Handle-Pull-v1": "Objective: Pull a handle up. Environment Details: The handle is positioned above the robot's end-effector. The robot must grasp and pull the handle upward. Handle positions are randomized.",
    "Handle-Press-Side-v1": "Objective: Press a handle down sideways. Environment Details: The handle is positioned for sideways pressing. The robot must apply force from the side to press the handle down. Handle positions are randomized.",
    "Handle-Pull-Side-v1": "Objective: Pull a handle up sideways. Environment Details: The handle is positioned for sideways pulling. The robot must grasp the handle from the side and pull upward. Handle positions are randomized.",
    "Stick-Push-v1": "Objective: Grasp a stick and push a box using the stick. Environment Details: The stick and box are positioned randomly. The robot must use the stick to apply force and push the box to a goal position.",
    "Stick-Pull-v1": "Objective: Grasp a stick and pull a box with the stick. Environment Details: The stick and box are positioned randomly. The robot must use the stick to apply force and pull the box to a goal position.",
    "Basketball-v1": "Objective: Dunk the basketball into the basket. Environment Details: The basketball and basket are positioned randomly. The robot must manipulate the basketball and place it into the basket.",
    "Soccer-v1": "Objective: Kick a soccer ball into the goal. Environment Details: The soccer ball and goal are positioned randomly. The robot must apply force to the ball to move it into the goal.",
    "Faucet-Open-v1": "Objective: Rotate the faucet counter-clockwise. Environment Details: The faucet is positioned randomly. The robot must apply force to rotate the faucet to the open position.",
    "Faucet-Close-v1": "Objective: Rotate the faucet clockwise. Environment Details: The faucet is positioned randomly. The robot must apply force to rotate the faucet to the closed position.",
    "Coffee-Push-v1": "Objective: Push a mug under a coffee machine. Environment Details: The mug and coffee machine are positioned randomly. The robot must move the mug into the correct position under the coffee machine.",
    "Coffee-Pull-v1": "Objective: Pull a mug from a coffee machine. Environment Details: The mug and coffee machine are positioned randomly. The robot must grasp the mug and pull it away from the machine.",
    "Coffee-Button-v1": "Objective: Push a button on the coffee machine. Environment Details: The coffee machine's button is positioned randomly. The robot must position its end-effector above the button and apply force to press it.",
    "Sweep-v1": "Objective: Sweep a puck off the table. Environment Details: The puck is positioned randomly on the table. The robot must apply lateral force to move the puck off the table.",
    "Sweep-Into-v1": "Objective: Sweep a puck into a hole. Environment Details: The puck is positioned randomly on the table near a hole. The robot must apply force to move the puck into the hole.",
    "Pick-Out-Of-Hole-v1": "Objective: Pick up a puck from a hole. Environment Details: The puck is positioned within a hole. The robot must grasp and lift the puck out of the hole. Puck and goal positions are randomized.",
    "Assembly-v1": "Objective: Pick up a nut and place it onto a peg. Environment Details: The nut and peg are positioned randomly. The robot must grasp the nut and place it accurately onto the peg.",
    "Shelf-Place-v1": "Objective: Pick and place a puck onto a shelf. Environment Details: The puck and shelf are positioned randomly. The robot must move the puck to the designated spot on the shelf.",
    "Push-Back-v1": "Objective: Pull a puck to a goal. Environment Details: The puck and goal are positioned randomly. The robot must apply force to move the puck back to the goal.",
    "Lever-Pull-v1": "Objective: Pull a lever down 90 degrees. Environment Details: The lever is positioned randomly. The robot must grasp and pull the lever to the required angle.",
    "Dial-Turn-v1": "Objective: Rotate a dial 180 degrees. Environment Details: The dial is positioned randomly. The robot must apply force to rotate the dial to the specified angle.",
    "Bin-Picking-v1": "Objective: Grasp the puck from one bin and place it into another bin. Environment Details: The puck and bins are positioned randomly. The robot must transfer the puck between bins.",
    "Box-Close-v1": "Objective: Grasp the cover and close the box with it. Environment Details: The box cover is positioned randomly. The robot must grasp and move the cover to close the box.",
    "Hand-Insert-v1": "Objective: Insert the gripper into a hole. Environment Details: The hole is positioned randomly. The robot must maneuver the gripper to align and insert it into the hole.",
    "Door-Lock-v1": "Objective: Lock the door by rotating the lock clockwise. Environment Details: The lock is positioned randomly. The robot must apply force to rotate the lock to the locked position.",
    "Door-Unlock-v1": "Objective: Unlock the door by rotating the lock counter-clockwise. Environment Details: The lock is positioned randomly. The robot must apply force to rotate the lock to the unlocked position."
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
