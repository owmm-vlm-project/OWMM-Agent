#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os,re
import textwrap
from typing import Dict, List, Optional, Tuple, Any

import imageio
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from habitat.core.logging import logger
from habitat.core.utils import try_cv2_import
from habitat.utils.common import flatten_dict
from habitat.utils.visualizations import maps
import PIL
cv2 = try_cv2_import()
def save_image(image, file_path):
    from PIL import Image
    img = Image.fromarray(image)
    img.save(file_path)

def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
):
    r"""Composites the foreground onto the background dealing with edge
    boundaries.
    Args:
        background: the background image to paste on.
        foreground: the image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: the image coordinates to paste the foreground.
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    Returns:
        The modified background image. This operation is in place.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]) : (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]) : (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        min_pad[0] : foreground.shape[0] - max_pad[0],
        min_pad[1] : foreground.shape[1] - max_pad[1],
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return background

    if mask is not None:
        mask = mask[
            min_pad[0] : foreground.shape[0] - max_pad[0],
            min_pad[1] : foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground
    return background


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    verbose: bool = True,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_")

    # File names are not allowed to be over 255 characters
    video_name_split = video_name.split("/")
    video_name = "/".join(
        video_name_split[:-1] + [video_name_split[-1][:251] + ".mp4"]
    )

    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    logger.info(f"Video created: {os.path.join(output_dir, video_name)}")
    if not verbose:
        images_iter: List[np.ndarray] = images
    else:
        images_iter = tqdm.tqdm(images)  # type: ignore[assignment]
    for im in images_iter:
        try:
            writer.append_data(im)
        except:
            continue
    writer.close()


def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view


def tile_images(render_obs_images: List[np.ndarray]) -> np.ndarray:
    """Tiles multiple images of non-equal size to a single image. Images are
    tiled into columns making the returned image wider than tall.
    """
    # Get the images in descending order of vertical height.
    render_obs_images = sorted(
        render_obs_images, key=lambda x: x.shape[0], reverse=True
    )
    img_cols = [[render_obs_images[0]]]
    max_height = render_obs_images[0].shape[0]
    cur_y = 0.0
    # Arrange the images in columns with the largest image to the left.
    col = []  # type: ignore
    for im in render_obs_images[1:]:
        # Check if the images in the same column have the same width
        if_same_width_in_the_same_col = True
        if len(col) > 0:
            if_same_width_in_the_same_col = col[-1].shape[1] == im.shape[1]

        if cur_y + im.shape[0] <= max_height and if_same_width_in_the_same_col:
            # Add more images on that column
            col.append(im)
            cur_y += im.shape[0]
        else:
            # Start to move to the next new column
            img_cols.append(col)
            col = [im]
            cur_y = im.shape[0]

    img_cols.append(col)
    col_widths = [max(col_ele.shape[1] for col_ele in col) for col in img_cols]
    # Get the total width of all the columns put together.
    total_width = sum(col_widths)

    # Tile the images, pasting the columns side by side.
    final_im = np.zeros(
        (max_height, total_width, 3), dtype=render_obs_images[0].dtype
    )
    cur_x = 0
    for i in range(len(img_cols)):
        next_x = cur_x + col_widths[i]
        total_col_im = np.concatenate(img_cols[i], axis=0)
        final_im[: total_col_im.shape[0], cur_x:next_x] = total_col_im
        cur_x = next_x
    return final_im


def observations_to_image(observation: Dict, info: Dict,
                          config: Any, frame_id: int, episode_id=0,stop_step = False) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().
        config: habitat configuration.
        frame_id: index of the frame being rendered.
        episode_id: (Optional) index of the episode being rendered.

    Returns:
        generated image of a single frame.
    """
    render_obs_images: List[np.ndarray] = []
    # TODO: for frames storage
    robot_names: Dict[
        Any, Any
    ] = {}
    for agent in config.habitat.simulator.agents:
        robot_names[agent] = (config.habitat.simulator.agents[agent].
                              articulated_agent_type)
    image_option = config.habitat_baselines.eval.image_option
    json_option = config.habitat_baselines.eval.json_option
    image_filter_list = config.habitat_baselines.eval.get("image_filter_list", ["head_rgb", "arm_workspace_rgb","nav_workspace_rgb"])
    image_dir = config.habitat_baselines.image_dir

    substrings = ["agent_0_has_finished_arm_action","agent_0_obj_pos","agent_0_target_pos",
                  "agent_0_localization_sensor","agent_0_ee_pos","ee_global_pos_sensor",
                  "agent_0_has_finished_oracle_nav","agent_0_robot_trans_martix",
                  "agent_0_obj_bounding_box","agent_0_target_bounding_box",
                  "agent_0_rec_bounding_box","agent_0_camera_extrinsic","agent_0_depth_inf","agent_0_arm_workspace_points",
                  "agent_0_detected_objects"]
    #"has_finished_oracle_nav"
    matched_data = {key: value.tolist() for key, value in observation.items() if any(sub in key for sub in substrings)}
    unload_name = ["robot_trans_martix","oracle_nav_target_path","camera_extrinsic","obj_bounding_box","target_bounding_box","rec_bounding_box","depth_inf",
    "depth_rot","depth_trans","arm_workspace_points","camera_matrix","depth_project","camera_info"]
    for sensor_name in observation:
        # if not arraylike, skip
        # print(sensor_name)
        if not hasattr(observation[sensor_name], "shape"):
            continue
        # print("sensor_name:",sensor_name)
        # if "arm_workspace_points" in sensor_name:
            # print("arm_workspace_points:",observation[sensor_name])
        if not any(sub in sensor_name for sub in unload_name):
            if len(observation[sensor_name].shape) > 1:
                obs_k = observation[sensor_name]
                if not isinstance(obs_k, np.ndarray):
                    obs_k = obs_k.cpu().numpy()
                if obs_k.dtype != np.uint8:
                    obs_k = obs_k * 255.0
                    obs_k = obs_k.astype(np.uint8)
                if obs_k.shape[2] == 1:
                    obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)
                render_obs_images.append(obs_k)

                def hit_key_list(name, key_list):
                    for k in key_list:
                        if k in name:
                            return True
                    return False
                sample_id = config.habitat_baselines.eval.episode_stored
                sample_frame = config.habitat_baselines.eval.episode_stored
                if "disk" in image_option and hit_key_list(sensor_name, image_filter_list):
                    assert image_dir is not None
                    image_ep_dir = os.path.join(image_dir, f"episode_{episode_id}")
                    if not os.path.exists(image_ep_dir):
                        os.makedirs(image_ep_dir)
                    temp = 0
                    if len(sample_frame) > 0:
                        for i in range(len(sample_frame)):
                            if int(episode_id) == int(sample_frame[i]['episode_id']):
                                temp = i
                                break
                        # print(f"temp:{temp}",flush = True)
                        for a in sample_frame[temp]['sample_frame']:
                            match = re.search(r"agent_(\d+)", sensor_name)
                            if match:
                                number = int(match.group(1))
                            if frame_id == a[0] and number == a[1] and int(episode_id) == int(sample_frame[temp]['episode_id']):
                                image_name = ('frame_' + str(frame_id) + '_' + sensor_name+
                                            robot_names[sensor_name[:7]] + sensor_name[7:])
                                # plt.imshow(obs_k)
                                # plt.axis('off')
                                # plt.savefig(os.path.join(image_ep_dir, image_name+'.png'),
                                #             bbox_inches='tight', pad_inches=0)
                                image_file_path = os.path.join(image_ep_dir, image_name+'.png')
                                save_image(obs_k, image_file_path)
                                break
                    else:
                        if sensor_name[:7] == "agent_0": #当存单agent的图片的时候
                            image_name = ('frame_' + str(frame_id) + '_' +sensor_name+
                                robot_names[sensor_name[:7]] + sensor_name[7:])
                            # print("obs:",obs_k.shape)
                            # plt.imshow(obs_k)
                            # plt.axis('off')
                            # plt.savefig(os.path.join(image_ep_dir, image_name+'.png'),
                            #             bbox_inches='tight', pad_inches=0)
                            image_file_path = os.path.join(image_ep_dir, image_name+'.png')
                            save_image(obs_k, image_file_path)
                image_ep_dir = os.path.join(image_dir, f"episode_{episode_id}")
                if not os.path.exists(image_ep_dir):
                        os.makedirs(image_ep_dir)
                # with open(os.path.join(image_ep_dir, 'frame_' + str(frame_id) +'_data.json'), 'w') as f:
                #     json.dump(matched_data, f, indent=2)
    if "disk" in json_option and stop_step is False:
        image_ep_dir = os.path.join(image_dir, f"episode_{episode_id}")
        if not os.path.exists(image_ep_dir):
            os.makedirs(image_ep_dir)
        if os.path.exists(os.path.join(image_ep_dir, 'sum_data.json')):
            with open(os.path.join(image_ep_dir, 'sum_data.json'), 'r') as f:
                data = json.load(f)
        else:
            data = {}
        if 'entities' not in data:
            data['entities'] = []
        config_data = {
            'step' : str(frame_id),
            'data' : matched_data
        }
        data['entities'].append(config_data)
        with open(os.path.join(image_ep_dir, 'sum_data.json'), 'w') as f:
            json.dump(data, f, indent=2)


    assert (
        len(render_obs_images) > 0
    ), "Expected at least one visual sensor enabled."

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1

    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)

    # draw collision
    collisions_key = "collisions"
    if collisions_key in info and info[collisions_key]["is_collision"]:
        render_frame = draw_collision(render_frame)

    top_down_map_key = "top_down_map"
    if top_down_map_key in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info[top_down_map_key], render_frame.shape[0]
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)
    return render_frame


def append_text_underneath_image(image: np.ndarray, text: str):
    """Appends text underneath an image of size (height, width, channels).

    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.

    :param image: The image to appends text underneath.
    :param text: The string to display.
    :return: A new image with text appended underneath.
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final


def overlay_text_to_image(
    image: np.ndarray, text: List[str], font_size: float = 0.5
):
    r"""Overlays lines of text on top of an image.

    First this will render to the left-hand side of the image, once that column is full,
    it will render to the right hand-side of the image.

    :param image: The image to put text on top.
    :param text: The list of strings which will be rendered (separated by new lines).
    :param font_size: Font size.
    :return: A new image with text overlaid on top.
    """
    h, w, c = image.shape
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    left_aligned = True
    for line in text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        if y > h:
            left_aligned = False
            y = textsize[1] + 10

        if left_aligned:
            x = 10
        else:
            x = w - (textsize[0] + 10)

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (0, 0, 0),
            font_thickness * 2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return np.clip(image, 0, 255)


def overlay_frame(frame, info, additional=None):
    """
    Renders text from the `info` dictionary to the `frame` image.
    """

    lines = []
    flattened_info = flatten_dict(info)
    for k, v in flattened_info.items():
        if isinstance(v, str):
            lines.append(f"{k}: {v}")
        elif isinstance(v, list):
            lines.append(f"{k}: {v[0]}")
        else:
            lines.append(f"{k}: {v:.2f}")
    if additional is not None:
        lines.extend(additional)

    frame = overlay_text_to_image(frame, lines, font_size=0.25)

    return frame
