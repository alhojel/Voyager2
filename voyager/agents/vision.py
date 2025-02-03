
    #    python -m voyager.agents.vision
import logging
import openai
import anthropic
import sys
sys.path.append('/Users/daisysong/Desktop/CS194agent/Voyager_OAI/voyager/')
#import voyager.utils as U
import utils as U
#from voyager import utils as U
import requests
import numpy as np
import os
import json
import base64
from datetime import datetime
# Import ChatOpenAI and HumanMessage from LangChain or the relevant library
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
#import pdb; pdb.set_trace()
from pathlib import Path


class VisionAgent:
    def __init__(
        self,
        #model_name="gpt-4-turbo", 
        model_name="gpt-4-mini", 
        # TODO 2: how to debug if this following vision_agent was called?
        temperature=0, 
        request_timeout=120,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
    ):
        print("\033[33mVisionAgent initialized\033[0m")  # Yellow color
        self.ckpt_dir = Path(ckpt_dir)
        self.chat_log = chat_log
        self.execution_error = execution_error

        U.f_mkdir(f"{ckpt_dir}/vision")
        self.vision_dir = self.ckpt_dir / "vision"
        #Define the vision directory using Path
        self.vision_dir = self.ckpt_dir / "vision"
        self.vision_dir.mkdir(parents=True, exist_ok=True)
        print(f"\033[34mEnsured directory exists: {self.vision_dir}\033[0m")  # Blue color for directory creation
        
        if resume:
            memory_file = self.vision_dir / "vision_memory.json"
            if memory_file.exists():
                print(f"\033[32mLoading Vision Agent from {memory_file}\033[0m")
                self.vision_memory = self._load_json(memory_file)
                print(f"\033[32mLoaded vision_memory.json successfully in vision.py.\033[0m")
            else:
                self.vision_memory = {}
                print(f"\033[31mvision_memory.json not found. Initializing empty vision_memory.\033[0m")  # Red color
        else:
            self.vision_memory = {}
            print("\033[33mInitialized empty vision_memory.\033[0m")  # Yellow color

        # if resume:
        #     print(f"\033[32mLoading Vision Agent from {ckpt_dir}/vision\033[0m")
        #     self.vision_memory = U.load_json(f"{ckpt_dir}/vision/vision_memory.json")
        #     print(f"\033[32mLoaded vision_memory.json successfully in vision.py.\033[0m")
        # else:
        #     self.vision_memory = {}
        #     print("\033[33mInitialized empty vision_memory.\033[0m")  # Yellow color


        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            max_tokens=2000
        )

    def _load_json(self, file_path):
        """Load JSON from a file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def _save_json(self, data, file_path):
        """Save JSON to a file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    # def update_vision_memory(self, vision_data):
    #     """
    #     Update vision memory with new data and save to file.

    #     Args:
    #         vision_data (dict): New vision data to add.
    #     """
    #     logging.info("Updating vision memory.")
    #     print(f"\033[33mVisionAgent.update_vision_memory called with: {vision_data}\033[0m")

    #     # Add a timestamp to the new data
    #     vision_data["timestamp"] = datetime.utcnow().isoformat()

    #     # Path to vision memory file
    #     memory_file_path = self.ckpt_dir / "vision" / "vision_memory.json"

    #     # Load existing memory if the file exists
    #     if memory_file_path.exists():
    #         self.vision_memory = self._load_json(memory_file_path)
    #     else:
    #         self.vision_memory = {}  # Ensure this is a list

    #     # Check if self.vision_memory is a list before appending
    #     if isinstance(self.vision_memory, list):
    #         # Append the new data to the vision memory
    #         self.vision_memory.append(vision_data)  # This line may cause the error if self.vision_memory is not a list
    #     else:
    #         logging.error("vision_memory is not a list. Current type: {}".format(type(self.vision_memory)))
    #         raise TypeError("vision_memory should be a list.")

    #     # Save the updated memory back to the JSON file
    #     self._save_json(self.vision_memory, memory_file_path)

    #     logging.info("New analysis added to vision_memory.json.")    
    def update_vision_memory(self, vision_data):
        """
        Update vision memory with new data and save to file.

        Args:
            vision_data (dict): New vision data to add.
        """
        logging.info("Updating vision memory.")
        print(f"\033[33mVisionAgent.update_vision_memory called with: {vision_data}\033[0m")

        # Add a timestamp to the new data
        timestamp = datetime.utcnow().isoformat()
        vision_data["timestamp"] = timestamp

        # Path to vision memory file
        #memory_file_path = self.ckpt_dir / "vision" / "vision_memory.json"
        memory_file_path = self.vision_dir / "vision_memory.json"

        # Load existing memory if the file exists
        if memory_file_path.exists():
            try:
                self.vision_memory = self._load_json(memory_file_path)
                if not isinstance(self.vision_memory, dict):
                    logging.error(f"Expected vision_memory.json to be a dict, but got {type(self.vision_memory)}")
                    raise TypeError("vision_memory.json should contain a dictionary.")
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON from {memory_file_path}: {e}")
                raise
        else:
            self.vision_memory = {}

        # Add the new data with the timestamp as the key
        self.vision_memory[timestamp] = vision_data

        # Save the updated memory back to the JSON file
        self._save_json(self.vision_memory, memory_file_path)

        logging.info("New analysis added to vision_memory.json.")
        print("\033[32mNew analysis added to vision_memory.json\033[0m")  # Green color


    # def analyze_image(self, image_path):
    #     """Send the image to GPT-4 Vision or similar models for analysis."""
    #     print(f"\033[33mVisionAgent.analyze_image called with: {image_path}\033[0m")

    #     # Function to encode the image
    #     def encode_image(image_path):
    #         with open(image_path, "rb") as image_file:
    #             return base64.b64encode(image_file.read()).decode('utf-8')

    #     # Encode the image to a base64 string
    #     base64_image = encode_image(image_path)

    #     # Define the response format
    #     response_format = """
    #     {
    #         "optimal_block": {
    #             "type": "string",
    #             "position": {
    #                 "x": "float",
    #                 "y": "float",
    #                 "z": "float"
    #             },
    #             "accessibility": "boolean"
    #         },
    #         "other_blocks": [
    #             {
    #                 "type": "string",
    #                 "position": {
    #                     "x": "float",
    #                     "y": "float",
    #                     "z": "float"
    #                 },
    #                 "accessibility": "boolean"
    #             }
    #         ],
    #         "spatial_reasoning": {
    #             "relative_positions": [
    #                 {
    #                     "block": "string",
    #                     "relative_position": {
    #                         "x": "float",
    #                         "y": "float",
    #                         "z": "float"
    #                     }
    #                 }
    #             ]
    #         }
    #     }
    #     """

    #     # Define the prompt
    #     prompt = f"""
    #     You are a highly capable assistant designed to analyze vision data and assist in completing any specified Minecraft task.
    #     Your role is to extract precise spatial insights from the provided visual data, enabling the AI bot to execute its tasks efficiently.

    #     ### Task Instructions
    #     1. **Enhanced Block Detection**:
    #     - Identify and determine the exact position (`x, y, z`) of blocks relevant to the task (e.g., `spruce_log`).
    #     - Prioritize the closest or most accessible block to the bot, providing coordinates for precise targeting.
    #     - If multiple blocks are detected, assess proximity, accessibility, and clustering to identify the optimal block for interaction.
    #     - Clearly describe the detected blocks, their positions, and any relevant contextual information in an organized format.

    #     2. **JSON Requirements**:
    #     - Return your output **strictly as a valid JSON object**.
    #     - Do not include any explanations, comments, or text outside the JSON structure.
    #     - The JSON must follow the format exactly as specified below.

    #     ### RESPONSE FORMAT:
    #     {response_format}

    #     Only respond with the JSON object. Do not include anything else in your output.
    #     """
        
    #    # Prepare messages using LangChain's BaseMessage classes
    #     messages = [
    #         HumanMessage(content=prompt),
    #         HumanMessage(content=f"Analyze the following image (base64 encoded): data:image/jpeg;base64,{base64_image}")
    #     ]

    #     #try:
    #         # Invoke the API
    #         # response = json.loads(self.llm.invoke(messages).content)
    #         # print(response)

    #         # # Handle the response
    #         # if isinstance(response, HumanMessage):
    #         #     response_data = response.content
    #         # elif isinstance(response, str):
    #         #     response_data = json.loads(response)  # Parse JSON if string
    #         # else:
    #         #     response_data = response  # Use as-is if already a dictionary

    #         # # Save the response to a JSON file
    #         # output_dir = os.path.join(self.ckpt_dir, "vision")
    #         # os.makedirs(output_dir, exist_ok=True)
    #         # print(f"\033[33mVisionAgent.analyze_image saving response to: {output_file_path}\033[0m")
    #         # output_file_path = os.path.join(output_dir, "vision_memory.json")

    #         # with open(output_file_path, 'w') as json_file:
    #         #     json.dump(response_data, json_file, indent=4)

    #         # return response_data
    #     try:
    #         # Invoke the model
    #         response = self.llm.invoke(messages).content
    #         response_data = json.loads(response)
    #         print(f"\033[33mVisionAgent.analyze_image response: {response_data}\033[0m")
    #         print(f"\033[33mVisionAgent.analyze_image response type: {type(response_data)}\033[0m")
            
    #         # Ensure that the expected fields are present and of the correct type
    #         if "optimal_block" in response_data:
    #             optimal_block = response_data["optimal_block"]
    #             if isinstance(optimal_block["position"], dict):
    #                 # Ensure position values are numeric
    #                 for key in ["x", "y", "z"]:
    #                     if not isinstance(optimal_block["position"].get(key), (int, float)):
    #                         raise ValueError(f"Expected numeric value for position '{key}', got {optimal_block['position'].get(key)}")

    #         response_data["image_path"] = image_path
    #         self.update_vision_memory(response_data)
    #         print(f"\033[33mVisionAgent.analyze_image updated vision_memory.json\033[0m")

    #         return response_data
    #     except Exception as e:
    #         raise RuntimeError(f"API call failed: {e}")     
    def analyze_image(self, image_path):
        """Send the image to GPT-4 Vision or similar models for analysis."""
        print(f"\033[33mVisionAgent.analyze_image called with: {image_path}\033[0m")

        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Encode the image to a base64 string
        base64_image = encode_image(image_path)

        # Define the response format
        response_format = """
        {
            "optimal_block": {
                "type": "string",
                "position": {
                    "x": "float",
                    "y": "float",
                    "z": "float"
                },
                "accessibility": "boolean"
            },
            "other_blocks": [
                {
                    "type": "string",
                    "position": {
                        "x": "float",
                        "y": "float",
                        "z": "float"
                    },
                    "accessibility": "boolean"
                }
            ],
            "spatial_reasoning": {
                "relative_positions": [
                    {
                        "block": "string",
                        "relative_position": {
                            "x": "float",
                            "y": "float",
                            "z": "float"
                        }
                    }
                ]
            }
        }
        """

        # Define the prompt
        prompt = f"""
        You are a highly capable assistant designed to analyze vision data and assist in completing any specified Minecraft task.
        Your role is to extract precise spatial insights from the provided visual data, enabling the AI bot to execute its tasks efficiently.

        ### Task Instructions
        1. **Enhanced Block Detection**:
        - Identify and determine the exact position (`x, y, z`) of blocks relevant to the task (e.g., `spruce_log`).
        - Prioritize the closest or most accessible block to the bot, providing coordinates for precise targeting.
        - If multiple blocks are detected, assess proximity, accessibility, and clustering to identify the optimal block for interaction.
        - Clearly describe the detected blocks, their positions, and any relevant contextual information in an organized format.

        2. **JSON Requirements**:
        - Return your output **strictly as a valid JSON object**.
        - Do not include any explanations, comments, or text outside the JSON structure.
        - The JSON must follow the format exactly as specified below.

        ### RESPONSE FORMAT:
        {response_format}

        Only respond with the JSON object. Do not include anything else in your output.
        """
        
       # Prepare messages using LangChain's BaseMessage classes
        messages = [
            HumanMessage(content=prompt),
            HumanMessage(content=f"Analyze the following image (base64 encoded): data:image/jpeg;base64,{base64_image}")
        ]

        try:
            # Invoke the model
            response = self.llm.invoke(messages).content
            response_data = json.loads(response)
            print(f"\033[33mVisionAgent.analyze_image response: {response_data}\033[0m")
            print(f"\033[33mVisionAgent.analyze_image response type: {type(response_data)}\033[0m")

             # Ensure that the expected fields are present and of the correct type
            if "optimal_block" in response_data:
                optimal_block = response_data["optimal_block"]
                if isinstance(optimal_block["position"], dict):
                    # Ensure position values are numeric
                    for key in ["x", "y", "z"]:
                        if isinstance(optimal_block["position"].get(key), str):
                            # Convert to float if the value is a string
                            optimal_block["position"][key] = float(optimal_block["position"][key])

            response_data["image_path"] = image_path
            self.update_vision_memory(response_data)
            print(f"\033[33mVisionAgent.analyze_image updated vision_memory.json\033[0m")

            return response_data
        except Exception as e:
            logging.error(f"API call failed: {e}")
            raise RuntimeError(f"API call failed: {e}")     

    def render_vision_message(self, image_path):
        #return HumanMessage(content=self.prompt)
        pass
    def process_vision_message(self, message):
        pass
    def get_insights(self, voxels, block_records, entities):
        pass
    def get_vision_memory(self):
        return self.vision_memory
# Example usage
# if __name__ == "__main__":
#     image_path = "/Users/daisysong/Desktop/CS194agent/Voyager_OAI/logs/visions/screenshot-2024-11-17T00-31-10-176Z.jpg"  # Update with path to the current image most recently captured
#     vision_agent = VisionAgent()
#     insights = vision_agent.analyze_image(image_path)
#     print("Vision Agent Insights:", insights)
#     # save the insights to a file
#     with open("vision_insights.txt", "w") as file:
#         file.write(insights)


# Roadmap Summary:
# Build a Hierarchical Multi-Agent System for efficient task allocation and execution.
# Incorporate Multi-Modal Perception for richer environmental understanding.
# Enable Dynamic Role Allocation to improve collaboration and adaptability.
# Add Long-Term Memory to leverage past experiences.
# Leverage LLMs for advanced planning and reasoning.
# Introduce Error Handling Mechanisms to improve robustness.
# Fine-Tune Vision Models for better spatial reasoning.
# Simulate Scalable Experiments to stress-test the system.
# Benchmark and Publish Results to validate your advancements.
# By implementing these steps, you can push the boundaries of multi-agent navigation systems and potentially achieve breakthroughs in embodied AI.

# TODO 3: Enhance Long-Term Memory and Knowledge Sharing
    # Current Challenge: Lack of memory limits agents’ ability to learn from past experiences.
    # Solution:
    # Add a multi-modal memory system to store successful strategies and environmental data (e.g., locations, paths, patterns).
    # Use retrieval-augmented generation (RAG) techniques to reference this memory during task execution.
    # Potential Breakthrough: Allows agents to reuse strategies and adapt to repetitive or complex tasks faster.


    # Use Large Language Models (LLMs) with Structured Fine-Tuning
    # Current Challenge: Limited reasoning capabilities in generating action plans.
    # Solution:
    # Fine-tune LLMs like GPT-4 to generate context-aware, multi-agent action plans.
    # Add specialized modules (e.g., planner, describer, critic) for interpreting visual cues and generating collaborative instructions.
    # Potential Breakthrough: Enhances planning and reasoning capabilities, making agents capable of solving unseen tasks effectively.

    # Introduce Proactive Error Handling and Adaptation
    # Current Challenge: Agents often fail when unexpected errors or environmental changes occur.
    # Solution:
    # Implement closed-loop feedback mechanisms to detect and correct errors autonomously.
    # Add a simulation layer for agents to test potential actions before execution.
    # Potential Breakthrough: Reduces task failure rates and increases the robustness of the system in unpredictable environments.

    # Optimize Vision Models for Navigation Tasks
    # Current Challenge: Vision models may lack the fine-tuned ability to detect and interact with Minecraft-specific objects.
    # Solution:
    # Train or fine-tune a vision transformer (e.g., Swin, CLIP) on Minecraft datasets.
    # Implement spatial reasoning modules to compute relative positions (x, y, z) and accessibility.
    # Potential Breakthrough: Enhances object detection and spatial reasoning, making agents more efficient in navigation.

    # Simulate Scalable Experiments
    # Current Challenge: Limited testing scenarios may not reveal the true potential or weaknesses of the system.
    # Solution:
    # Use procedurally generated Minecraft environments with varying levels of complexity.
    # Simulate large-scale collaboration tasks with 10+ agents to test scalability and adaptability.
    # Potential Breakthrough: Demonstrates real-world viability and provides insights for further optimization.

    # Publish and Benchmark Against State-of-the-Art
    # Current Challenge: It's unclear how your system compares to existing frameworks like HAS, Voyager, and STEVE.
    # Solution:
    # Conduct benchmarks on open-ended navigation tasks (e.g., map exploration, multi-modal goal search) using your enhanced system.
    # Publish results comparing your system's efficiency, success rates, and scalability.
    # Potential Breakthrough: Establishes credibility in the research community and identifies areas for improvement.
