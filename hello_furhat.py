from furhat_remote_api import FurhatRemoteAPI

# 'localhost' means we are looking for the robot on THIS computer.
# If you were using a real robot, you would put its IP address here.
furhat = FurhatRemoteAPI("localhost")

# 1. Set the voice (Furhat SDK comes with several English voices)
furhat.set_voice(name='Alex')

# 2. Say something
print("Sending command to Furhat...")
furhat.say(text="Hello! I am ready to be your Dungeon Master.")

# 3. Perform a gesture
furhat.gesture(name="BigSmile")