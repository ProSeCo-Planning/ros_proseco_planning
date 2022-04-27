#!/usr/bin/env python3
import os
import carla
import queue
import random
import argparse
import numpy as np
import json
from glob import glob
import pygame
from carla import ColorConverter as cc
from pygame.locals import K_ESCAPE
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT

# Scenes
SCENES = [
    # Two-lanes-different-direction
    ("Town01", 130.0, 133.43 + 1.75, False),  # 1
    ("Town01", 130.0, 199.14 + 1.75, False),  # 2
    ("Town01", 130.0, 330.59 + 1.75, False),  # 3
    ("Town01", 170.0, 59.49 + 1.75, False),  # 4
    ("Town01", 180.0, 1.96 + 1.75, False),  # 5
    ("Town01", -2.03 - 1.75, 20.0, True),  # 6
    ("Town01", 392.33 - 1.75, 50.0, True),  # 7
    # Two-lanes-same-direction
    ("Town04", -17.0 - 1.75, -220.0, True),  # 8
    ("Town04", -16.0 - 1.75, 70.0, True),  # 9
]

# Colors for agents
COLORS = [
    "0, 0, 0",
    "255, 0, 0",
    "0, 255, 0",
    "0, 0, 255",
    "255, 255, 0",
    "255, 0, 255",
    "0, 255, 255",
    "255, 255, 255",
]


class ActorSpectator(object):
    def __init__(self, world, args):
        self.world = world
        self.sensor = None
        self.queue = queue.Queue()
        self.width = args.width
        self.height = args.height
        self.fov = args.fov
        self.location = args.location
        self.rotation = args.rotation
        self.surface = None
        self.index = 0
        self.recording = args.recording
        self.record_path = args.record_path
        self.file_num = 0

        if not os.path.exists(os.path.expanduser(self.record_path)) and self.recording:
            os.makedirs(os.path.expanduser(self.record_path))

        self.get_actors()
        self.init_pygame()
        self.init_blueprint()
        self.set_camera(self.index)

    def get_actors(self):
        """Finds all agents"""
        self.actors = [
            actor
            for actor in self.world.get_actors().filter("vehicle.*")
            if "Agent" in actor.attributes["role_name"]
        ]
        self.actors = sorted(self.actors, key=lambda x: x.id)

    def init_pygame(self):
        """Initializes the pygame window"""
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

    def init_blueprint(self):
        """Initializes the camera blueprint"""
        self.bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.bp.set_attribute("image_size_x", str(self.width))
        self.bp.set_attribute("image_size_y", str(self.height))
        self.bp.set_attribute("fov", str(self.fov))

    def set_camera(self, index):
        """Sets the camera sensor"""
        index = index % len(self.actors)

        if self.sensor is not None:
            self.sensor.destroy()
            self.surface = None

        self.sensor = self.world.spawn_actor(
            self.bp,
            carla.Transform(
                carla.Location(
                    x=self.location[0], y=self.location[1], z=self.location[2]
                ),
                carla.Rotation(
                    yaw=self.rotation[0], pitch=self.rotation[1], roll=self.rotation[2]
                ),
            ),
            attach_to=self.actors[index],
            attachment_type=carla.AttachmentType.Rigid,
        )

        self.sensor.listen(self.queue.put)
        self.index = index

    def render(self, frame):
        """Renders a spectator window"""
        # Render camera images
        array = self._retrieve_data(frame)
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.surface is not None:
            self.display.blit(self.surface, (0, 0))

        pygame.display.flip()

        # Save pygame display if enabled
        if self.recording:
            self.file_num += 1
            filename = os.path.join(
                os.path.expanduser(self.record_path), "image_%04d.png" % self.file_num
            )
            pygame.image.save(self.display, filename)

    def parse_events(self):
        """Parse the keyboard inputs"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 1
            elif event.type == pygame.KEYUP:
                if event.key == K_ESCAPE:
                    return 1
        if any(x != 0 for x in pygame.key.get_pressed()):
            self.parse_keys(pygame.key.get_pressed())
            return 2

    def parse_keys(self, keys):
        """Controls the camera focus"""
        prev_index = self.index
        if keys[K_RIGHT]:
            self.index += 1
        if keys[K_LEFT]:
            self.index -= 1

        if prev_index != self.index:
            self.set_camera(self.index)

    def make_video(self, fps):
        """Makes a mp4 file from the recorded images and deletes images afterwards"""
        path = os.path.expanduser(self.record_path)
        images = os.path.join(path, "image_%04d.png")
        cmd = "cd {} && ffmpeg -framerate {} -i {} -pix_fmt yuv420p video.mp4".format(
            path, fps, images
        )
        os.system(cmd)
        for image in glob(path + "/image_*.png"):
            os.remove(image)

    def _retrieve_data(self, frame):
        """Returns the image data"""
        while True:
            try:
                image = self.queue.get()
                if image.frame == frame:
                    self.image = self._preprocess_data(image)
                    return self.image
            except:
                return self.image

    def _preprocess_data(self, image):
        """Process and returns the image data"""
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def destroy(self):
        """Destroys the camera sensor and quits pygame"""
        self.sensor.destroy()
        pygame.quit()


class MCTSVisualizer(object):
    def __init__(self, args, simulate_physics=False):

        self.num_agents = 0
        self.num_obstacles = 0
        self.horizon = 0
        self.obstacles = []
        self.agents = dict()
        self.simulate_physics = simulate_physics
        self.recording = args.recording
        self.scene = SCENES[args.scene - 1]
        self.reverse = self.scene[-1]
        self.offset = self.scene[1:-1][::-1] if self.reverse else self.scene[1:-1]
        self.y_sign = 1.0 if self.reverse else -1.0
        self.timeout = 2.0

        # Parse scenario file
        self.parse_json(args.config)
        # Prepare the world
        self.init_world(args.host, args.port)
        # Make scenario
        self.make_scenario()
        # Prepare Spectator
        self.spectator = ActorSpectator(self.world, args)

    def parse_json(self, file_path):
        """Extracts information to reconstruct the scenario"""
        with open(file_path, "r") as f:
            self.trajectory_dict = json.load(f)
        self.horizon = len(self.trajectory_dict["agents"][0]["trajectory"])
        self.dt = self.trajectory_dict["agents"][0]["trajectory"][1]["time"]

    def init_world(self, host, port):
        """Prepares the world"""
        client = carla.Client(host, port)
        if client.get_world().get_map().name != self.scene[0]:
            client.set_timeout(100.0)
            self.world = client.load_world(self.scene[0])
            client.set_timeout(self.timeout)
        else:
            self.world = client.get_world()

        self.settings = self.world.get_settings()
        _ = self.world.apply_settings(
            carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=self.dt)
        )

        # Sets blueprints for agents and obstacles
        self.bp_agent = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        self.bp_obstacle = self.world.get_blueprint_library().find(
            "vehicle.volkswagen.t2"
        )

    def make_scenario(self):
        """Spawns agents and obstacles"""

        # Obstacles
        for obstacle in self.trajectory_dict["obstacles"] or []:
            # Get obstacle spawn point
            x = obstacle["position_x"] + self.offset[0]
            y = self.y_sign * obstacle["position_y"] + self.offset[1]
            yaw = -180.0 * obstacle["heading"] / np.pi
            # Reverse x and y if scene is following y-axis
            if self.reverse:
                x, y = y, x
                yaw += 90.0
            # Set color
            self.bp_obstacle.set_attribute("role_name", "Obstacle")
            color = random.choice(
                self.bp_obstacle.get_attribute("color").recommended_values
            )
            self.bp_obstacle.set_attribute("color", color)
            # Set spawn point and disable physics
            transform = carla.Transform()
            transform.location.x = x
            transform.location.y = y
            transform.location.z = 0.05
            transform.rotation.yaw = yaw
            obstacle = self.world.spawn_actor(self.bp_obstacle, transform)
            obstacle.set_simulate_physics(self.simulate_physics)
            # Store obstacle
            self.obstacles.append(obstacle)

        # Agents
        for index, agent in enumerate(self.trajectory_dict["agents"] or []):
            # Get agent's trajectory
            traj_x = [
                traj["position_x"] + self.offset[0] for traj in agent["trajectory"]
            ]
            traj_y = [
                self.y_sign * traj["position_y"] + self.offset[1]
                for traj in agent["trajectory"]
            ]
            traj_h = [-180.0 * traj["heading"] / np.pi for traj in agent["trajectory"]]
            # Reverse x and y if scene is following y-axis
            if self.reverse:
                traj_x, traj_y = traj_y, traj_x
                traj_h += 90.0
            # Set color
            self.bp_agent.set_attribute("role_name", "Agent_" + str(index))
            color = COLORS[index]
            self.bp_agent.set_attribute("color", color)
            # Set spawn point and disable physics
            transform = carla.Transform()
            transform.location.x = traj_x[0]
            transform.location.y = traj_y[0]
            transform.location.z = 0.05
            transform.rotation.yaw = traj_h[0]
            agent = self.world.spawn_actor(self.bp_agent, transform)
            agent.set_simulate_physics(self.simulate_physics)
            # Store agent's trajectory
            self.agents[agent] = dict(x=traj_x, y=traj_y, yaw=traj_h)

        self.frame = self.world.tick()

    def run(self):
        """Runs the scenario"""

        for t in range(self.horizon):
            for agent, trajectory in self.agents.items():
                transform = carla.Transform()
                transform.location.x = trajectory["x"][t]
                transform.location.y = trajectory["y"][t]
                transform.rotation.yaw = trajectory["yaw"][t]
                agent.set_transform(transform)

            self.frame = self.world.tick()
            self.render(self.frame)

        if self.recording:
            record_fps = 1 / self.dt
            self.spectator.make_video(record_fps)

    def render(self, frame):
        """Renders the scenario"""
        e = self.spectator.parse_events()
        if e == 1:
            return exit()
        elif e == 2:
            self.frame = frame = self.world.tick()
        self.spectator.render(frame)

    def clear(self):
        """Clears the scenario from obstacles and agents"""
        actor_list = self.obstacles + [agent for agent in self.agents.keys()]
        for actor in actor_list:
            actor.destroy()

    def close(self):
        """Deletes the entire scenario entities as well as the spectator"""
        # Clear scenario
        self.clear()
        # Destroy spectator
        self.spectator.destroy()
        # Reset server settings
        self.world.apply_settings(self.settings)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="MCTS Visualization")
    argparser.add_argument(
        "-config",
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "trajectory_annotated.json"
        ),
        type=str,
        help="Path to scenario file (.json)",
    )
    argparser.add_argument(
        "-host",
        metavar="HOST",
        default="localhost",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-port",
        metavar="PORT",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "-scene",
        metavar="SCENE",
        default=1,
        type=int,
        help="Location where the scenario is running (default: 1)",
    )
    argparser.add_argument(
        "-resolution",
        metavar="WIDTHxHEIGHT",
        default="1280x720",
        help="window resolution (default: 1280x720)",
    )
    argparser.add_argument(
        "-fov",
        metavar="FOV",
        default=100.0,
        type=float,
        help="Field of camera view (default: 100.0)",
    )
    argparser.add_argument(
        "-location",
        metavar="LOCATION",
        nargs="+",
        default=[-16.0, 0.0, 12.0],
        type=float,
        help="Position of the camera (x, y, z) (default: -16.0 0.0 12.0)",
    )
    argparser.add_argument(
        "-rotation",
        metavar="ROTATION",
        nargs="+",
        default=[0.0, -30.0, 0.0],
        type=float,
        help="Rotation of the camera (yaw, pitch, roll) (default: 0.0 -30.0 0.0)",
    )
    argparser.add_argument(
        "--recording", action="store_true", help="Enables recording the images"
    )
    argparser.add_argument(
        "-record_path",
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "MCTS_CarlaViz"
        ),
        help="Directory to save images if recording is on",
    )

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.resolution.split("x")]
    visualizer = MCTSVisualizer(args)
    try:
        visualizer.run()
    finally:
        visualizer.close()
