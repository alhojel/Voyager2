import os.path
import time
import warnings
from typing import SupportsFloat, Any, Tuple, Dict

import requests
import json

import gymnasium as gym
from gymnasium.core import ObsType

import voyager.utils as U

from .minecraft_launcher import MinecraftInstance
from .process_monitor import SubprocessMonitor


class VoyagerEnv(gym.Env):
    def __init__(
        self,
        mc_port=None,
        azure_login=None,
        server_host="http://127.0.0.1",
        server_port=3000,
        viewer_port=None,
        username=None,
        request_timeout=600,
        log_path="./logs",
        pause_on_think=True,
    ):
        if not mc_port and not azure_login:
            raise ValueError("Either mc_port or azure_login must be specified")
        if mc_port and azure_login:
            warnings.warn(
                "Both mc_port and mc_login are specified, mc_port will be ignored"
            )
        self.mc_port = mc_port
        self.azure_login = azure_login
        self.server = f"{server_host}:{server_port}"
        self.server_port = server_port
        self.viewer_port = viewer_port
        self.username = username
        self.request_timeout = request_timeout
        self.log_path = log_path
        self.mineflayer = self.get_mineflayer_process(server_port)
        self.pause_on_think = pause_on_think

        if azure_login:
            self.mc_instance = self.get_mc_instance()
        else:
            self.mc_instance = None
        self.has_reset = False
        self.reset_options = None
        self.connected = False
        self.server_paused = False

    def get_mineflayer_process(self, server_port):
        U.f_mkdir(self.log_path, "mineflayer")
        file_path = os.path.abspath(os.path.dirname(__file__))
        return SubprocessMonitor(
            commands=[
                "node",
                U.f_join(file_path, "mineflayer/index.js"),
                str(server_port),
            ],
            name="mineflayer",
            ready_match=r"Server started on port (\d+)",
            log_path=U.f_join(self.log_path, "mineflayer"),
        )

    def get_mc_instance(self):
        print("Creating Minecraft server")
        U.f_mkdir(self.log_path, "minecraft")
        return MinecraftInstance(
            **self.azure_login,
            mineflayer=self.mineflayer,
            log_path=U.f_join(self.log_path, "minecraft"),
        )

    def check_process(self):
        if self.mc_instance and not self.mc_instance.is_running:
            # if self.mc_instance:
            #     self.mc_instance.check_process()
            #     if not self.mc_instance.is_running:
            print("Starting Minecraft server")
            self.mc_instance.run()
            self.mc_port = self.mc_instance.port
            self.reset_options["port"] = self.mc_instance.port
            print(f"Server started on port {self.reset_options['port']}")
        retry = 0
        while not self.mineflayer.is_running:
            print("Mineflayer process has exited, restarting")
            self.mineflayer.run()
            if not self.mineflayer.is_running:
                if retry > 3:
                    raise RuntimeError("Mineflayer process failed to start")
                else:
                    continue
            print(self.mineflayer.ready_line)
            payload = self.reset_options.copy()
            if self.viewer_port is not None:
                payload["viewerPort"] = self.viewer_port
            if self.username is not None:
                payload["username"] = self.username

            res = requests.post(
                f"{self.server}/start",
                #json=self.reset_options,
                json=payload,
                timeout=self.request_timeout,
            )
            if res.status_code != 200:
                print(f"Response content: {res.content}")
                self.mineflayer.stop()
                raise RuntimeError(
                    #f"Minecraft server reply with code {res.status_code}"
                    f"Minecraft server replied with code {res.status_code}"
                )
            return res.json()

    def step(
        self,
        code: str,
        programs: str = "",
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if not self.has_reset:
            raise RuntimeError("Environment has not been reset yet")
        self.check_process()
        #self.unpause()
        data = {
            "code": code,
            "programs": programs,
        }
        res = requests.post(
            f"{self.server}/step", json=data, timeout=self.request_timeout
        )
        if res.status_code != 200:
            raise RuntimeError("Failed to step Minecraft server")
        returned_data = res.json()
        #self.pause()
        return json.loads(returned_data)

    def render(self):
        raise NotImplementedError("render is not implemented")

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if options is None:
            options = {}

        if options.get("inventory", {}) and options.get("mode", "hard") != "hard":
            raise RuntimeError("inventory can only be set when options is hard")

        self.reset_options = {
            "port": self.mc_port,
            "reset": options.get("mode", "hard"),
            "inventory": options.get("inventory", {}),
            "equipment": options.get("equipment", []),
            "spread": options.get("spread", False),
            "waitTicks": options.get("wait_ticks", 5),
            "position": options.get("position", None),
            "viewerPort": self.viewer_port,
            "username": self.username,
        }

        #self.unpause()
        self.mineflayer.stop()
        time.sleep(1)  # wait for mineflayer to exit

        returned_data = self.check_process()
        self.has_reset = True
        self.connected = True
        # All the reset in step will be soft
        self.reset_options["reset"] = "soft"
        #self.pause()
        return json.loads(returned_data)

    def close(self):
        #self.unpause()
        if self.connected:
            res = requests.post(f"{self.server}/stop")
            if res.status_code == 200:
                self.connected = False
        if self.mc_instance:
            self.mc_instance.stop()
        self.mineflayer.stop()
        return not self.connected

    def pause(self):
        if not self.pause_on_think:
            return False
        if self.mineflayer.is_running and not self.server_paused:
            res = requests.post(f"{self.server}/pause")
            if res.status_code == 200:
                self.server_paused = True
        return self.server_paused

    def unpause(self):
        if not self.pause_on_think:
            return False
        if self.mineflayer.is_running and self.server_paused:
            res = requests.post(f"{self.server}/pause")
            if res.status_code == 200:
                self.server_paused = False
            else:
                print(res.json())
        return self.server_paused

# The `bridge.py` file defines a class called `VoyagerEnv`, which is a custom environment for interacting with a Minecraft server using the Gymnasium library. This environment is likely part of a larger project that involves training agents to perform tasks in a Minecraft setting. Here's a breakdown of its key components and functionality:

# ### Key Components:

# 1. **Imports:**
#    - The file imports various modules, including standard libraries (`os`, `time`, `warnings`, `json`), third-party libraries (`requests`, `gymnasium`), and local utility functions from `voyager.utils`.

# 2. **Class Definition:**
#    - The `VoyagerEnv` class inherits from `gym.Env`, making it compatible with the Gymnasium framework for reinforcement learning.

# ### Constructor (`__init__` method):
# - **Parameters:**
#   - The constructor takes several parameters, including `mc_port`, `azure_login`, server configuration options, and logging options.
# - **Initialization:**
#   - It initializes instance variables based on the provided parameters, checks for required parameters, and sets up a subprocess to run a Mineflayer server (a Node.js library for creating Minecraft bots).

# ### Key Methods:

# 1. **`get_mineflayer_process`:**
#    - This method starts a Mineflayer process using a subprocess monitor, which manages the lifecycle of the Mineflayer server and logs its output.

# 2. **`get_mc_instance`:**
#    - This method is responsible for creating a Minecraft server instance. It likely interacts with the Minecraft server to set it up for use.

# 3. **`reset`:**
#    - This method resets the environment, allowing for a new episode to start. It handles options for resetting the server, such as inventory and equipment settings.

# 4. **`close`:**
#    - This method stops the environment and cleans up resources, including stopping the Mineflayer process and the Minecraft instance.

# 5. **`pause` and `unpause`:**
#    - These methods control the state of the server, allowing it to be paused or unpaused based on the `pause_on_think` flag.

# 6. **`render`:**
#    - This method is defined but not implemented, indicating that rendering functionality is not currently available in this environment.

# ### Summary:
# Overall, `bridge.py` serves as a bridge between a reinforcement learning agent and a Minecraft server, allowing the agent to interact with the game environment programmatically. It manages the setup, reset, and control of the Minecraft server and the Mineflayer bot, facilitating the training of agents in a Minecraft setting.