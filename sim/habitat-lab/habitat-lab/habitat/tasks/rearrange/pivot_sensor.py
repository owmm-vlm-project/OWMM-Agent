import json
from gym import spaces
from dataclasses import dataclass

from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface


def matrx4_to_list(obj):
    empty_list = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            empty_list[i][j] = obj[i][j]
    return empty_list


@registry.register_sensor
class CameraInfoSensor(UsesArticulatedAgentInterface, Sensor):
    """Sensor to camera info from simulator"""
    cls_uuid: str = "camera_info"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self.depth_sensor_name = "head_depth"
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return CameraInfoSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Text(max_length=10000)

    def get_observation(self, *args, **kwargs):
        if self.agent_id is not None:
            sensor_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
        else:
            sensor_name = self.depth_sensor_name

        render_camera = self._sim._sensors[sensor_name]._sensor_object.render_camera
        viewport = render_camera.viewport
        projection_matrix = render_camera.projection_matrix
        camera_matrix = render_camera.camera_matrix
        projection_size = render_camera.projection_size()[0]

        camera_info = {
            "viewport": viewport,
            "projection_matrix": matrx4_to_list(projection_matrix),
            "camera_matrix": matrx4_to_list(camera_matrix),
            "projection_size": projection_size,
        }
        camera_info_str = json.dumps(camera_info)
        return camera_info_str


@registry.register_sensor
class PIVOTInfoSensor(UsesArticulatedAgentInterface, Sensor):
    """Sensor to get information for visual prompting"""
    cls_uuid: str = "pivot_info"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return PIVOTInfoSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Text(max_length=10000)

    def get_observation(self, *args, **kwargs):
        pivot_info = {}
        for i in range(len(self._sim.ep_info.target_receptacles[0])-1):
            pivot_info[f"recep_any_targets|{i}"] = self._sim.ep_info.target_receptacles[0][i]
        for i in range(len(self._sim.ep_info.goal_receptacles[0])-1):
            pivot_info[f"TARGET_any_targets|{i}"] = self._sim.ep_info.goal_receptacles[0][i]
        for key, val in self._sim.ep_info.info['object_labels'].items():
            pivot_info[val] = key

        pivot_info_str = json.dumps(pivot_info)
        return pivot_info_str


def get_pivot_info_sensors(sim, *args, **kwargs):
    lab_sensors_config = kwargs.get("lab_sensors_config", None)

    if lab_sensors_config is None:
        lab_sensors_config = {}
        @dataclass
        class PIVOTInfoSensorConfig:
            uuid: str = "pivot_info"
            type: str = "PIVOTInfoSensor"
        @dataclass
        class CameraInfoSensorConfig:
            uuid: str = "camera_info"
            type: str = "CameraInfoSensor"
            depth_sensor_name: str = "head_depth"

        lab_sensors_config["pivot_info"] = PIVOTInfoSensorConfig()
        lab_sensors_config["camera_info"] = CameraInfoSensorConfig()

    return {
        "pivot_info": PIVOTInfoSensor(
            sim, lab_sensors_config["pivot_info"], *args, **kwargs
        ),
        "camera_info": CameraInfoSensor(
            sim, lab_sensors_config["camera_info"], *args, **kwargs
        )
    }


def get_pivot_sim_info(sim, *args, **kwargs):
    pivot_sensors = get_pivot_info_sensors(sim, *args, **kwargs)
    pivot_info = {
        sensor_name: sensor.get_observation()
        for sensor_name, sensor in pivot_sensors.items()
    }
    return pivot_info
