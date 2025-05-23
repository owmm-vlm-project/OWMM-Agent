import matplotlib.pyplot as plt
import os
import numpy as np
import json
import re,shutil
import base64
import requests
import time,random
from PIL import Image
from io import BytesIO
import openai
from openai import OpenAI
from torchvision import transforms
import torch
from habitat_mas.pivot.run_pivot import run_pivot
json_format_info = {
    "reasoning":"Based on the current image information and history, think and infer the actions that need to be executed and action's information.",
    "action":"The action name of your reasoning result.",
    "action_information":"If the action is \"nav_to_point\",\"pick\" or \"place\", the information format should be \"<box>[[x1, y1, x2, y2]]</box>\" to indicate location information,where x1, y1, x2, y2 are the coordinates of the bounding box;if the action is \"search_scene_frame\",the information format should be \"id\",where id is the value of the frame index you choose.",
    "summarization":"summarize based on the current action information and history."
}
pivot_json_format_info = {
    "reasoning":"Based on the current image information and history, think and infer the actions that need to be executed and action's information.",
    "action":"The action name of your reasoning result.",
    "action_information":"If the action is \"nav_to_point\",\"pick\" or \"place\", the information format should be \"<box>[[x1, y1, x2, y2]]</box>\" to indicate location information,where x1, y1, x2, y2 are the coordinates of the bounding box;if the action is \"search_scene_frame\",the information format should be \"id\",where id is the value of the frame index you choose.",
    "summarization":"summarize based on the current action information and history."
}
def save_image(image, file_path):
    from PIL import Image
    img = Image.fromarray(image)
    img.save(file_path)
class PIVOTAgent:
    def __init__(self,**kwargs):
        self.robot_history = None
        self.agent_name = kwargs.get("agent_name", "agent_0")
        self.client = OpenAI(api_key='123', base_url='http://0.0.0.0:23333/v1')
        self.prepare_action_num = 3
        self.rgb_image_store_num = 0
        self.debug_item = 1
        self.cal_output_it = 0
        self.search_action_information = -1
        self.attempt_search = 0
        self.pre_action_name = ""
        self.rag_image_name_list_set = [
'target_rec.png', 'goal_rec.png',
'random_scene_graph_2.png', 'random_scene_graph_3.png',
'random_scene_graph_4.png', 'random_scene_graph_5.png',
'random_scene_graph_6.png', 'random_scene_graph_7.png']
        self.prepare_action = [{
                    "name": "reset_arm",
                    "arguments": ['agent_0']
                },
                {
                    "name": "wait",
                    "arguments": ['1','agent_0']
                }]
        self.use_pivot = True

    def _2d_to_3d_single_point(self, depth_obs, depth_rot,depth_trans,pixel_x, pixel_y):
        # depth_camera = self._sim._sensors[depth_name]._sensor_object.render_camera

        # hfov = float(self._sim._sensors[depth_name]._sensor_object.hfov) * np.pi / 180.
        # W, H = depth_camera.viewport[0], depth_camera.viewport[1]
        W = 512
        H = 512
        hfov = 1.5707963267948966
        # Intrinsic matrix K
        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0., 1, 0],
            [0., 0., 0, 1]
        ])

        # Normalize pixel coordinates
        xs = (2.0 * pixel_x / (W - 1)) - 1.0  # normalized x [-1, 1]
        ys = 1.0 - (2.0 * pixel_y / (H - 1))  # normalized y [1, -1]

        # Depth value at the pixel
        depth = depth_obs[0,pixel_y,pixel_x,0]

        # Create the homogeneous coordinates for the pixel in camera space
        xys = np.array([xs * depth, ys * depth, -depth, 1.0]).reshape(4, 1)

        # Apply the inverse of the intrinsic matrix to get camera space coordinates
        xy_c = np.matmul(np.linalg.inv(K), xys)

        # Get the rotation and translation of the camera
        depth_rotation = np.array(depth_rot)
        depth_translation = np.array(depth_trans)

        # Get camera-to-world transformation
        T_world_camera = np.eye(4)
        T_world_camera[0:3, 0:3] = depth_rotation
        T_world_camera[0:3, 3] = depth_translation

        # Apply transformation to get world coordinates
        T_camera_world = np.linalg.inv(T_world_camera)
        points_world = np.matmul(T_camera_world, xy_c)

        # Get non-homogeneous points in world space
        points_world = points_world[:3, :] / points_world[3, :]
        return points_world.flatten()
    def get_episode_prompt(self,data_path):
        data_dir_path = os.path.dirname(data_path)
        data_dir_num = os.path.basename(data_dir_path)
        prompt_json_path = os.path.join(os.path.dirname(data_dir_path),'task_prompt.json')
        with open(prompt_json_path, 'r') as file:
            data = json.load(file)
        task_description = next((item["task_description"] for item in data if item["image_number"] == str(data_dir_num)), None)
        assert task_description,"OH NO!CAN NOT FIND CURRENT TASK DESCRIPTION"
        return task_description
    def process_prompt(self,task_prompt,history = None):
        if not history:
            robot_history = ""
        else:
            robot_history = f"Robot's history:\"{history}\""
        image_token_pad = '<IMAGE_TOKEN>'
        # set json format for pivot
        json_format = pivot_json_format_info if self.use_pivot else json_format_info
        question_prompt_align_with_ovmm = f"""You are an AI visual assistant that can manage a single robot. You receive the robot's task,one image representing the robot's current view and eight frames of the scene from the robot's tour. You need to output the robot's next action. Actions the robot can perform are "search_scene_frame","nav_to_point","pick" and "place".
These frames are from the robot's tour of the scene:
Image-1:{image_token_pad}
Image-2:{image_token_pad}
Image-3:{image_token_pad}
Image-4:{image_token_pad}
Image-5:{image_token_pad}
Image-6:{image_token_pad}
Image-7:{image_token_pad}
Image-8:{image_token_pad}
If you can not find the target you need to identify,you should find the frame that the robot should navigate to complete the task,and output "search_scene_frame" action and the id of frame.
Robot's current view is: Image-9:{image_token_pad}.If you can find the position that you should navigate to, pick or place,you should output your action information.In robot's current view, some green points may appear,indicating the positions that the robot's arm can reach. When there are green points on the object that needs to be picked, it means the robot's arm can pick up the object. When there are enough green points on the goal container where the object needs to be placed, it means the robot's arm can place the object into the goal container.
Besides,you need to explain why you choose this action in your output and summarize by combining your chosen action with historical information.
Robot's Task: {task_prompt}{robot_history}Your output format should be in pure JSON format as follow:{json_format}."""
        return question_prompt_align_with_ovmm
    def process_message(self,data_path,robot_image,prompt,output_path): #sr:send&receive
        content = [
            {
                "type":"text",
                "text": prompt
            },
        ]
        data_dir_path = os.path.dirname(data_path)
        seed_num = int(os.path.basename(data_dir_path))
        rag_image_path_list = [os.path.join(data_dir_path,item)
                               for item in self.rag_image_name_list_set]
        random.seed(seed_num)
        random.shuffle(rag_image_path_list)
        # print(output_path)
        for rag_image_path in rag_image_path_list:
            # with open(rag_image_path,"rb") as rag_image:
            #     temp_image_info = {
            #         "type": "image_url",
            #         "image_url": {
            #             "url": f"data:image/png;base64,{base64.b64encode(rag_image.read()).decode('utf-8')}",

            #         }
            #     }
            #     content.append(temp_image_info)
            CLOUD_URL = ""
            cloud_url = os.path.join(CLOUD_URL,rag_image_path.split('/', 2)[2])
            temp_image_info = {
                "type": "image_url",
                "image_url": {
                    "url": cloud_url,
                }
            }
            content.append(temp_image_info)
        robot_image_PIL = Image.fromarray(np.squeeze(robot_image))
        robot_image_buffered = BytesIO()
        robot_image_PIL.save(robot_image_buffered,format = 'PNG')
        robot_image_info = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(robot_image_buffered.getvalue()).decode('utf-8')}",
                # "url": f"data:image/png;base64,{base64.b64encode(robot_image_buffered.getvalue()).decode('utf-8')}",
            }
        }
        content.append(robot_image_info)
        return content,rag_image_path_list
    def query_and_receive(self,content,client):

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                'role':'user',
                'content': content
            }],
            temperature=0.0,
            top_p=0.5,
            response_format={"type":"json_object"})
        return response
    def process_nav_point(self,depth_obs, depth_rot,depth_trans,pixel_xy):
        pixel_x, pixel_y = pixel_xy
        point_3d = self._2d_to_3d_single_point(depth_obs.cpu(), depth_rot.cpu(),depth_trans.cpu(),pixel_x, pixel_y)
        IGNORE_NODE = [-100]
        point_3d_debug = np.concatenate((point_3d, IGNORE_NODE))
        # print("1:point_3d_debug:",point_3d_debug)
        # point_3d_debug = point_3d_debug[0]
        # print("2:point_3d_debug",point_3d_debug)
        # point_3d_debug = point_3d_debug.tolist()
        # print("3:point_3d_debug",point_3d_debug)
        return point_3d_debug
    def extract_gptoutput_to_json(self,input_string):
        print("BEFORE_extract:",input_string)
        # pattern = r'["\'](reasoning|action|action_information|summarization)["\']\s*:\s*["\'](.*?)["\'],'
        
        # matches = re.findall(pattern, input_string)
        
        # extracted_data = {field: value for field, value in matches}
        extracted_data = json.loads(input_string)
        # new_json = json.dumps(extracted_data, ensure_ascii=False, indent=4)
        
        return extracted_data
    def process_action(self,action,action_information,data_path,observations,rgb_image_path_list,output_path):
        if action == "search_scene_frame":
            if (self.pre_action_name == action) and (str(self.search_action_information) == str(action_information)):
                if self.attempt_search > 2:
                    raise ValueError("Output more than 2 same search action information")
            data_dir_path = os.path.dirname(data_path)
            metadata_json_path = os.path.join(data_dir_path,'metadata.json')
            # print("rgb_image_path_list:",rgb_image_path_list)
            basename_list = [os.path.basename(item)
                             for item in rgb_image_path_list]
            print("basename_list:",basename_list)
            print("action_information_sc:",action_information)
            rgb_image_path = rgb_image_path_list[int(action_information)-1]
            rgb_image_name = os.path.basename(rgb_image_path)
            with open(metadata_json_path, 'r') as file:
                data = json.load(file)
            for item in data:
                if rgb_image_name in item["obs_files"]:
                    nav_position = item["position"]
            assert nav_position,"Code has bug"
            if (self.pre_action_name == action) and (str(self.search_action_information) == str(action_information)):
                self.attempt_search += 1
            else:
                self.attempt_search = 0
            self.search_action_information = action_information
            self.pre_action_name = action
            return {
                "name": "nav_to_position",
                "arguments": {
                    "target_position": nav_position,
                    "robot": self.agent_name,
                }
            }
        if action == "nav_to_point":
            print("ACTIONINFORMATION:",action_information)
            target_position = self.process_nav_point(observations['depth_obs'],
                                                  observations['depth_rot'],observations['depth_trans'],action_information),
            print("target_position:",target_position)
            target_position = target_position[0]
            target_position = target_position.tolist()
            self.pre_action_name = action
            return {
                "name": "nav_to_position",
                "arguments": {
                    "target_position":target_position,
                    "robot": self.agent_name,
                }
            }
        if (action == "pick") or (action == "place"):
            print("ACTIONINFORMATION:",action_information)
            robot_armworkspace_image = observations["arm_workspace_rgb"].cpu().numpy()
            robot_third_image = observations["third_rgb"].cpu().numpy()
            save_image(np.squeeze(robot_armworkspace_image.copy()),f"{output_path}/robot_output_{action}_{self.cal_output_it}.png")
            save_image(np.squeeze(robot_third_image.copy()),f"{output_path}/third_robot_output_{action}_{self.cal_output_it}.png")
            self.cal_output_it+=1
            self.pre_action_name = action
            return {
                "name": f"{action}_key_point",
                "arguments": {
                    "position": action_information,
                    "robot": self.agent_name,
                }
            }
        self.pre_action_name = 'wait'
        return {
            "name": f"wait",
            "arguments": ['3000',self.agent_name]
        }
    def process_vlm_output(self,data_path,vlm_output,rgb_image_path_list,observations,output_path,action_information=None):
        # print("vlm_output:",vlm_output)
        print("vlm_output:",vlm_output)
        vlm_action = vlm_output['action']
        if self.use_pivot:
            vlm_action_information = action_information
        else:
            vlm_action_information = vlm_output['action_information']
        vlm_action_summarization = vlm_output['summarization']
        vlm_return = self.process_action(
            vlm_action,vlm_action_information,data_path,observations,rgb_image_path_list,output_path
        )
        return vlm_return,vlm_action_summarization
    def make_robopoint_query(self,query):
        action_name = query["action"]
        task_prompt = query["task_prompt"]
        robot_history = "" if not query["robot_history"] else f"Robot's history: {query['robot_history']}"
        task_def = f"""The robot need to {task_prompt}.Now the robot need to {action_name}.{robot_history}"""
        task_def += f"Find a few spots for robot to execute the next action."
        task_def += f"""Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above.The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."""
        return task_def
    def robopoint_model(self,im,query):
        image_PIL = Image.fromarray(im) #input_shape:(512, 512, 3)
        image_buffered = BytesIO()
        image_PIL.save(image_buffered,format = 'PNG')
        print("QUERY_robopoint:",self.make_robopoint_query(query))
        prompt_seq = [self.make_robopoint_query(query), image_buffered]
        content = []
        for elem in prompt_seq:
            if isinstance(elem, str):
                content.append({'type': 'text', 'text': elem})
            elif isinstance(elem, BytesIO):
                base64_image_str = base64.b64encode(elem.getvalue()).decode('utf-8')
                image_url = f'data:image/jpeg;base64,{base64_image_str}'
                content.append(
                    {'type': 'image_url', 'image_url': {'url': image_url}})
        messages = [{'role': 'user', 'content': content}]
        client_robopoint = OpenAI(api_key='123', base_url='http://0.0.0.0:23333/v1')
        model_name_robopoint = client_robopoint.models.list().data[0].id
        response = client_robopoint.chat.completions.create(
            model=model_name_robopoint,
            messages=messages,
            temperature=0.0,
            top_p=0.5
        )
        output_text = response.choices[0].message.content
        print("robopoint_return:",output_text)
        try:
            prediction_str = output_text.strip(" []")
            prediction_points = [tuple(map(float, point.strip("()").split(", "))) for point in prediction_str.split("), (")]
            prediction_center = np.mean(prediction_points, axis=0)
            prediction_center_scaled = [int(prediction_center[0]*512),int(prediction_center[1]*512)]
        except:
            raise ValueError("Wrong robopoint output")
        return prediction_center_scaled
    def vlm_eval_response(self,observations,data_path, output_path,camera_info=None):
        if self.prepare_action_num:
            action_num = self.prepare_action_num
            self.prepare_action_num -= 1
            return self.prepare_action[2-action_num]
        self.prepare_action_num = 2
        #this is for action cycle(ensure reset&wait before and action)
        robot_image = observations["head_rgb"].cpu().numpy()
        # save_image(np.squeeze(robot_image.copy()),f"./eval_in_sim_info/{os.path.basename(os.path.dirname(data_path))}/robot_input_{self.rgb_image_store_num}.png")
        # print("robot_image:",robot_image.shape)
        self.rgb_image_store_num+=1
        task_prompt = self.get_episode_prompt(data_path)
        prompt = self.process_prompt(task_prompt,history = self.robot_history)
        content, rgb_image_path_list = self.process_message(data_path,robot_image,prompt,output_path)
        # print("query_content:",content)
        print("system_ask:",prompt)
        API_KEY = ""
        BASE_URL = ""
        openai.api_key = API_KEY
        openai.base_url = BASE_URL
        vlm_output = self.query_and_receive(content,openai)
        camera_info = {
            'viewport': np.array([512, 512]),
            'projection_matrix': np.array(observations['depth_project'].cpu()[0]), 
            'camera_matrix': np.array(observations['camera_matrix'].cpu()[0]),
            'projection_size': 2.0}
        # TODO: Query pivot
        if self.use_pivot:
            openai_key = API_KEY
            openai_base_url = BASE_URL
            # print("vlm_output:",vlm_output)
            text_vlm_output = self.extract_gptoutput_to_json(vlm_output.choices[0].message.content)
            print("text_vlm_output:",text_vlm_output)
            # if self.debug_item == 0:
            #     text_vlm_output = {
            #         'reasoning': "The robot's task is to move the cracker box from the Home library to the Central countertop. Based on the robot's history, it was previously directed to search in frame 2 to locate the cracker box. The current view (Image-9) show the cracker box, indicating that the robot needs to navigate closer to search for it.",
            #         'action': 'nav_to_point', 
            #         'action_information': '[[0,0,512,512]]', 
            #         'summarization': 'The robot will nav closer to pick the cracker box.'
            #         }
            # if text_vlm_output["action"] == "search_scene_frame" and self.debug_item == 1:
            #     action_information = text_vlm_output["action_information"]
            #     self.debug_item = 0
            if text_vlm_output["action"] == "search_scene_frame":
                action_information = text_vlm_output["action_information"]
            else:
                text_vlm_output["task_prompt"] = task_prompt
                text_vlm_output["robot_history"] = self.robot_history
                USE_PIVOT = False
                if USE_PIVOT:
                    # print("shape:",np.squeeze(robot_image).shape) #(512, 512, 3)
                    action_information = run_pivot(
                                    im=np.squeeze(robot_image),
                                    query=text_vlm_output,
                                    n_samples_init=10,
                                    n_samples_opt=6,
                                    n_iters=2,
                                    n_parallel_trials=1,
                                    openai_api_key=openai_key,
                                    openai_base_url = openai_base_url,
                                    camera_info=camera_info,
                                )
                else:
                    #use robopoint
                    action_information = self.robopoint_model(
                        im=np.squeeze(robot_image),
                        query=text_vlm_output,
                    )
            vlm_return,self.robot_history = self.process_vlm_output(
                data_path,text_vlm_output,rgb_image_path_list,observations,output_path,action_information
            )
        else:
            raise ValueError("Pivot is not implemented")
            vlm_return, self.robot_history = self.process_vlm_output(
                data_path, vlm_output, rgb_image_path_list, observations,
            )
        return vlm_return
