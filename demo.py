from voyager import Voyager
import os
from dotenv import load_dotenv
import threading  # Import Python's threading module
import time

# Load environment variables from .env file (if you're using one)
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No API key found. Please set the OPENAI_API_KEY environment variable.")

# You can also use mc_port instead of azure_login, but azure_login is highly recommended
# mc_port is only for local testing and will not work in the cloud.
# azure_login = {
#     "client_id": os.getenv("AZURE_CLIENT_ID", "default_client_id"),
#     "redirect_url": "https://127.0.0.1/auth-response",
#     "secret_value":os.getenv("AZURE_SECRET_VALUE", "default_secret_value"),  # Fetch from env,
#     "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
# }
azure_login = {
    "client_id": "3ba45205-c9c0-46bc-8216-4b481f55873d",
    "redirect_url": "https://127.0.0.1/auth-response",
    "secret_value": "",
    "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
}

bot_0 = Voyager(
    bot_id=0,  # Unique ID for the first bot
    mc_port=55377,
    #azure_login=azure_login,
    openai_api_key=openai_api_key,
    max_iterations=100,
    #ckpt_dir="./ckpt",
)

bot_1 = Voyager(
    bot_id=1,  # Unique ID for the second bot
    mc_port=55377,  # Ensure it connects to the same Minecraft server
    #azure_login=azure_login,
    openai_api_key=openai_api_key,
    max_iterations=100,
    #ckpt_dir="./ckpt",
)

def run_bot(voyager):
    try:
        voyager.learn(reset_env=False)  
    except Exception as e:
        print(f"Error: {str(e)}")

bot1_thread = threading.Thread(target=run_bot, args=(bot_0,))
bot2_thread = threading.Thread(target=run_bot, args=(bot_1,))
bot1_thread.start()
bot2_thread.start()
bot1_thread.join()
bot2_thread.join()

# bot1_thread.connect()
# bot2_thread.connect()
# spawn_location = (100, 64, 100)  # Example coordinates
# bot1_thread.move_to(spawn_location)
# bot2_thread.move_to(spawn_location)

"""
{
    "OPENAI_API_KEY": "",
    "OPENAI_ORG_ID": "",
    "GEMINI_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "REPLICATE_API_KEY": "",
    "GROQCLOUD_API_KEY": "",
    "HUGGINGFACE_API_KEY": "",
    "QWEN_API_KEY":""
}
"""


# Function to run a bot and capture vision
# def run_bot(bot_id, mc_port, openai_api_key, output_dir):
#     # Create a unique directory for the bot's vision data
#     bot_output_dir = os.path.join(output_dir, f"bot_{bot_id}")
#     os.makedirs(bot_output_dir, exist_ok=True)

#     # Initialize the bot
#     bot = Voyager(
#         bot_id=bot_id,
#         mc_port=mc_port,
#         openai_api_key=openai_api_key,
#         server_port=3000 + bot_id,  # Unique server port for each bot
#         bot_username=f"bot_{bot_id}",  # Unique username for each bot
#         resume=True,
#     )

# # Main function to run two bots concurrently
# def run_two_bots():
#     mc_port = 25565  # Minecraft server port
#     openai_api_key = openai_api_key
#     output_dir = "./bot_visions"

#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)

#     # Run two bots in separate threads
#     bot_1_thread = threading.Thread(target=run_bot, args=(0, mc_port, openai_api_key, output_dir))
#     bot_2_thread = threading.Thread(target=run_bot, args=(1, mc_port, openai_api_key, output_dir))

#     bot_1_thread.start()
#     bot_2_thread.start()

#     bot_1_thread.join()
#     bot_2_thread.join()

#     print("Both bots have completed their tasks.")

# # Run the main function
# if __name__ == "__main__":
#     run_two_bots()