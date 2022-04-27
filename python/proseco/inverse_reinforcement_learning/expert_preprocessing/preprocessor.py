from .frenetSystem import FrenetSystem
import xml.etree.ElementTree as xml
import pyproj
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from ros_proseco_planning.msg import (
    CostParam,
    NoiseParam,
    Trajectory,
    Control,
    Features,
    AgentVec,
    Agent,
    Action,
    ScenarioInfo,
    State,
)
import pickle
import os


class LL2XYProjector:
    "Source: Interaction Code"

    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = (
            math.floor((lon_origin + 180.0) / 6) + 1
        )  # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(proj="utm", ellps="WGS84", zone=self.zone, datum="WGS84")
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]


class AgentMsgGenerator:
    def __init__(self, agent_index):
        """
        Class to create a ros agent message which is part of the ros trajectory message (see msg folder)

        Arguments:
            agent_index int - index of the agent for which the message is generated
        """
        self.agent_index = agent_index

    def initializeNewMsg(self):
        """
        Initializes new ros feature and state messages
        """
        self.features = Features()
        self.state = State()

    def add_state(self, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y):
        """
        Adds all necessary state variables to the ros state message
        """
        self.state.posX = pos_x
        self.state.posY = pos_y
        self.state.velX = vel_x
        self.state.velY = vel_y
        self.state.accX = acc_x
        self.state.accY = acc_y

    def add_state_features(
        self,
        diff_vel_vel_des,
        abs_lane_diff,
        diff_des_lane_cent,
        desired_vel,
        collided,
        invalidState,
        desired_lane,
    ):
        """
        Adds all necessary state dependend variables to the ros feature message
        """
        self.features.diff_vel_vel_des = diff_vel_vel_des
        self.features.abs_lane_diff = abs_lane_diff
        self.features.diff_des_lane_cent = diff_des_lane_cent
        self.features.desired_vel = desired_vel
        self.features.collided = collided
        self.features.invalidState = invalidState
        self.features.desiredLane = desired_lane

    def add_action_features(self, invalidAction, absoluteAverageAccY):
        """
        Adds all necessary action dependend variables to the ros feature message
        """
        self.features.invalidAction = invalidAction
        self.features.averageAbsoluteAccY = absoluteAverageAccY

    def get_agent_msg(self):
        """
        Retrieves the agent message
        """
        msg = Agent()
        msg.id = self.agent_index
        msg.features = self.features
        msg.action = Action()
        msg.state = self.state
        agent_vec = AgentVec()
        agent_vec.agentVec = [msg]
        return agent_vec


class Preprocessor(object):
    def __init__(
        self,
        number_of_lanes,
        lane_width,
        number_of_agents,
        automated_setting_of_desired_vel,
        automated_setting_of_desired_lane,
    ):
        """
        Class which produces from a interaction dataset .csv file and a reference.osm file a trajectory_annotated.csv file and pickle files containing
        ros trajectory messages. These can be used as expert trajectories in the irl procedure

        Arguments:
            number_of_lanes int - number of lanes in the scenario
            lane_width float - lane width in the scenario
            number_of_agent int - number of agents present in the scenario
            automated_setting_of_desired_vel bool - flag if desired vel should set to the veloctiy at the beginning of the observed trajectory
            automated_setting_of_desired_lane bool - flag if desired lane should set to the lane at the beginning of the observed trajectory
        """

        self.number_of_lanes = number_of_lanes
        self.lane_width = lane_width
        self.system = FrenetSystem()
        self.projector = LL2XYProjector(0.0, 0.0)
        self.agents_dataframe = pd.DataFrame()
        self.number_of_agents = number_of_agents
        self.obstacles = []
        self.desired_velocities = []
        self.desired_lanes = []
        self.stage_time = 800
        self.scenario_name = "scenario_1"
        self.automated_setting_of_desired_vel = automated_setting_of_desired_vel
        self.automated_setting_of_desired_lane = automated_setting_of_desired_lane
        self.number_of_stages = 6

    def set_desired_vel(self, desired_velocities):
        self.desired_velocities = desired_velocities

    def set_desired_lanes(self, desired_lanes):
        self.desired_lanes = desired_lanes

    def add_obstacle(self, frenet_x, frenet_y, length, width, heading):
        """
        Add obstacle to the virtual scenario (in frenet coordinates) - is used to create the obstacle columns in trajectory_annotated.csv
        """
        obstacle = frenet_x, frenet_y, length, width, heading
        self.obstacles.append(obstacle)

    def load_reference_lane(self, osm_file):
        """
        Extracts a curve which is given in the osm_file and sets it as reference curve for the frenet coordinate system

        Arguments:
            osm_file string - full path to osm file
        """
        xml_root = xml.parse(osm_file).getroot()
        unsorted_curve = []
        for node in xml_root.findall("node"):
            point = self.projector.latlon2xy(
                float(node.get("lat")), float(node.get("lon"))
            )
            unsorted_curve.append(point)

        def first_element(elem):
            return elem[0]

        unsorted_curve.sort(key=first_element)
        curve = unsorted_curve
        curve = self.smooth_curve(curve)
        self.system.add_reference_curve(curve)

    def smooth_curve(self, curve):
        """
        Takes a curve (list of tuples of (x,y) coordinates) and returns a cubic smoothed version

        Arguments:
            curve [(x,y)] - list of tuples parameterizing a curve

        Return:
            [(x,y)] - smoothed curve
        """
        x_coordinates = []
        y_coordinates = []
        for point in curve:
            x, y = point
            x_coordinates.append(x)
            y_coordinates.append(y)
        x_new = np.linspace(
            x_coordinates[0], x_coordinates[-1], num=20000, endpoint=True
        )
        f2 = interp1d(x_coordinates, y_coordinates, kind="cubic")
        f_new = f2(x_new)
        new_curve = []
        for i in range(len(f_new)):
            point = x_new[i], f_new[i]
            new_curve.append(point)
        return new_curve

    def load_input_csv(self, csv_file):
        """
        Load input interaction dataset .csv file

        Arguments:
            csv_file String - full path to the csv file from the interaction dataset
        """
        self.agents_dataframe = pd.read_csv(csv_file)

    def get_frenet_coordinates_for_agent(self, agent_index):
        """
        Returns frenet coordinate of a specific agent with respect to the specified curve which is the reference curve of the frenet coordinate system

        Arguments:
            agent_index int - index of the agent -> extracts cartesian coordinates of agent from agent_data_frame (coming from the interaction dataset .csv)

        Return:
            frenet_x_coordinates [float] - list of floats with the frenet x values of the trajectory of the agent
            frenet_y_coordinates [float] - list of floats with the frenet y values of the trajectory of the agent
            timestamp_list [float] - list with time steps of the (x,y) positions of the trajectory

        """
        df_agent = self.agents_dataframe[
            self.agents_dataframe["track_id"] == agent_index
        ]
        print(df_agent.head())
        x_sequence = df_agent["x"].to_numpy()
        y_sequence = df_agent["y"].to_numpy()
        time_sequence = df_agent["timestamp_ms"].to_numpy()
        frenet_x_coordinates = []
        frenet_y_coordinates = []
        timestamp_list = []
        for i in range(len(x_sequence)):
            point = x_sequence[i], y_sequence[i]
            x_frenet, y_frenet = self.system.project_xy_to_frenet(point)
            if x_frenet is not None:
                frenet_x_coordinates.append(x_frenet)
                frenet_y_coordinates.append(y_frenet)
                timestamp_list.append(time_sequence[i])
            else:
                if i >= len(x_sequence) - 1:
                    break
                print("None appeared - switch to average")
                point_t_1 = x_sequence[i - 1], y_sequence[i - 1]
                point_t_plus_1 = x_sequence[i + 1], y_sequence[i + 1]
                x_frenet_1, y_frenet_1 = self.system.project_xy_to_frenet(point_t_1)
                x_frenet_plus_1, y_frenet_plus_1 = self.system.project_xy_to_frenet(
                    point_t_plus_1
                )
                assert x_frenet_1 is not None
                assert x_frenet_plus_1 is not None
                x_frenet = 0.5 * x_frenet_1 + 0.5 * x_frenet_plus_1
                y_frenet = 0.5 * y_frenet_plus_1 + 0.5 * y_frenet_1
                frenet_x_coordinates.append(x_frenet)
                frenet_y_coordinates.append(y_frenet)
                timestamp_list.append(time_sequence[i])

        return frenet_x_coordinates, frenet_y_coordinates, timestamp_list

    def create_output_files(
        self, output_csv_file_path, output_pickle_file_path, pickle_file_prefix
    ):
        """
        Main method which creates the output file trajectory_annotated.csv and the pickle files (each for every agent) containing the ros trajectory message

        Arguments:
            output_csv_file_path string - complete path to output file (exclusive file name)
            output_pickle_file_path string - subdir containing the pickle files
            pickle_file_prefix string - prefix for all pickle files (that are than numbered after the agent indexes)
        """

        ##ofAgents,stage,time,m_id,m_positionX,m_positionY,m_lane,m_positionYDesired,m_velocityX,m_velocityDesiredX,m_velocityY,m_accelerationX,m_accelerationY,m_heading,m_totalAcceleration,m_steeringAngle,m_egoReward,m_coopReward,m_width,m_length,m_id,m_positionX,m_positionY,m_lane,m_positionYDesired,m_velocityX,m_velocityDesiredX,m_velocityY,m_accelerationX,m_accelerationY,m_heading,m_totalAcceleration,m_steeringAngle,m_egoReward,m_coopReward,m_width,m_length,m_id,m_positionX,m_positionY,m_lane,m_positionYDesired,m_velocityX,m_velocityDesiredX,m_velocityY,m_accelerationX,m_accelerationY,m_heading,m_totalAcceleration,m_steeringAngle,m_egoReward,m_coopReward,m_width,m_length,#ofObstacles,#ofLanes,laneWidth,
        agent_dfs = []

        ######### Create own df for each agent ###############
        for i in range(self.number_of_agents):
            df_agent = pd.DataFrame()
            vehicle_length, vehicle_width = self.extract_constants(i + 1)
            (
                frenet_x_coordinates,
                frenet_y_coordinates,
                timestamp_list,
            ) = self.get_frenet_coordinates_for_agent(i + 1)
            velocities_x = self.numeric_velocity_calculation(frenet_x_coordinates, 0.1)
            velocities_y = self.numeric_velocity_calculation(frenet_y_coordinates, 0.1)
            acceleration_x = self.numeric_acceleration_calculation(
                frenet_x_coordinates, 0.1
            )
            acceleration_y = self.numeric_acceleration_calculation(
                frenet_y_coordinates, 0.1
            )
            lane_vec = self.calculate_lane(frenet_y_coordinates[1:-1])
            if self.automated_setting_of_desired_vel:
                self.desired_velocities.append(velocities_x[0])
            if self.automated_setting_of_desired_lane:
                self.desired_lanes.append(lane_vec[0])
            desired_velocity = self.desired_velocities[i]
            desired_y_coordinate = self.lane_center_y_coordinate(self.desired_lanes[i])

            df_agent["time"] = timestamp_list[1:-1]
            length = len(df_agent["time"])
            df_agent["m_id" + str(i)] = np.repeat(i, length)
            df_agent["m_positionX" + str(i)] = frenet_x_coordinates[1:-1]
            df_agent["m_positionY" + str(i)] = frenet_y_coordinates[1:-1]
            df_agent["m_lane" + str(i)] = lane_vec
            df_agent["m_positionYDesired" + str(i)] = np.repeat(
                desired_y_coordinate, length
            )
            df_agent["m_velocityX" + str(i)] = velocities_x[:-1]
            df_agent["m_velocityY" + str(i)] = velocities_y[:-1]
            df_agent["m_velocityDesiredX" + str(i)] = np.repeat(
                desired_velocity, length
            )
            df_agent["m_accelerationX" + str(i)] = acceleration_x
            df_agent["m_accelerationY" + str(i)] = acceleration_y
            df_agent["m_width" + str(i)] = np.repeat(vehicle_width, length)
            df_agent["m_length" + str(i)] = np.repeat(vehicle_length, length)
            df_agent["m_heading" + str(i)] = np.repeat(0.0, length)
            df_agent["m_totalAcceleration" + str(i)] = np.repeat(0.0, length)
            df_agent["m_steeringAngle" + str(i)] = np.repeat(0.0, length)
            df_agent["m_egoReward" + str(i)] = np.repeat(0.0, length)
            df_agent["m_coopReward" + str(i)] = np.repeat(0.0, length)
            agent_dfs.append(df_agent)
        agent_dfs = self.filter_time_stamp(agent_dfs)
        new_agent_dfs = []

        ######## Calculate stages ###################
        for df in agent_dfs:
            df["stage"] = self.calculated_stages(df["time"].to_numpy())
            df["time"] = df["time"].to_numpy() / 1000.0
            new_df = df.reset_index(drop=True)
            new_df = new_df[new_df["stage"] <= self.number_of_stages]
            print(new_df.head())
            new_agent_dfs.append(new_df)

        ######## Create Feature lists for agents ###########
        # feature_lists =[]
        for i in range(self.number_of_agents):
            trajectory_msg = self.create_trajectory_msg(new_agent_dfs[i], i)

            self.save_trajectory_msg(
                trajectory_msg=trajectory_msg,
                agent_index=i,
                pickle_file_path=output_pickle_file_path,
                pickle_prefix=pickle_file_prefix,
            )

        ########### Merge agent dataframes #################
        main_df = new_agent_dfs[0]
        for df in new_agent_dfs[1:]:
            del df["time"]
            del df["stage"]
            for col_name in df.columns:
                main_df[col_name] = df[col_name]
        length = len(main_df["time"])
        main_df["#ofAgents"] = np.repeat(self.number_of_agents, length)
        main_df["#ofObstacles"] = np.repeat(len(self.obstacles), length)
        main_df["#ofLanes"] = np.repeat(self.number_of_lanes, length)
        main_df["laneWidth"] = np.repeat(self.lane_width, length)
        obstacle_index = 0

        ########### Add obstacles to main datadrame ############
        # o_id,o_distanceX,o_distanceY,o_length,o_width,o_heading
        for obstacle in self.obstacles:
            o_x, o_y, o_length, o_width, o_heading = obstacle
            main_df["o_id" + str(obstacle_index)] = np.repeat(obstacle_index, length)
            main_df["o_distanceX" + str(obstacle_index)] = np.repeat(o_x, length)
            main_df["o_distanceY" + str(obstacle_index)] = np.repeat(o_y, length)
            main_df["o_length" + str(obstacle_index)] = np.repeat(o_length, length)
            main_df["o_width" + str(obstacle_index)] = np.repeat(o_width, length)
            main_df["o_heading" + str(obstacle_index)] = np.repeat(o_heading, length)
            obstacle_index += 1

        ########## Reorder column names ####################
        main_df = main_df[self.get_ordered_cols()]
        # print(main_df.head())
        main_df.columns = self.get_final_column_name()
        # print(main_df.head())

        ######### Save df ###################
        main_df.to_csv(
            os.path.join(output_csv_file_path, "trajectory_annotated.csv"),
            index=False,
        )

    def get_ordered_cols(self):
        """
        creates the col names as they should be present in 'trajectory_annotated.csv'
        """
        col_names = ["#ofAgents", "stage", "time"]
        for i in range(self.number_of_agents):
            col_names.append("m_id" + str(i))
            col_names.append("m_positionX" + str(i))
            col_names.append("m_positionY" + str(i))
            col_names.append("m_lane" + str(i))
            col_names.append("m_positionYDesired" + str(i))
            col_names.append("m_velocityX" + str(i))
            col_names.append("m_velocityDesiredX" + str(i))
            col_names.append("m_velocityY" + str(i))
            col_names.append("m_accelerationX" + str(i))
            col_names.append("m_accelerationY" + str(i))
            col_names.append("m_heading" + str(i))
            col_names.append("m_totalAcceleration" + str(i))
            col_names.append("m_steeringAngle" + str(i))
            col_names.append("m_egoReward" + str(i))
            col_names.append("m_coopReward" + str(i))
            col_names.append("m_width" + str(i))
            col_names.append("m_length" + str(i))
        col_names.append("#ofObstacles")
        for i in range(len(self.obstacles)):
            col_names.append("o_id" + str(i))
            col_names.append("o_distanceX" + str(i))
            col_names.append("o_distanceY" + str(i))
            col_names.append("o_length" + str(i))
            col_names.append("o_width" + str(i))
            col_names.append("o_heading" + str(i))
        col_names.append("#ofLanes")
        col_names.append("laneWidth")
        return col_names

    def get_final_column_name(self):
        """
        creates the corrected col names as they should be present in 'trajectory_annotated.csv'
        """
        col_names = ["#ofAgents", "stage", "time"]
        for i in range(self.number_of_agents):
            col_names.append("m_id")
            col_names.append("m_positionX")
            col_names.append("m_positionY")
            col_names.append("m_lane")
            col_names.append("m_positionYDesired")
            col_names.append("m_velocityX")
            col_names.append("m_velocityDesiredX")
            col_names.append("m_velocityY")
            col_names.append("m_accelerationX")
            col_names.append("m_accelerationY")
            col_names.append("m_heading")
            col_names.append("m_totalAcceleration")
            col_names.append("m_steeringAngle")
            col_names.append("m_egoReward")
            col_names.append("m_coopReward")
            col_names.append("m_width")
            col_names.append("m_length")
        col_names.append("#ofObstacles")
        for i in range(len(self.obstacles)):
            col_names.append("o_id")
            col_names.append("o_distanceX")
            col_names.append("o_distanceY")
            col_names.append("o_length")
            col_names.append("o_width")
            col_names.append("o_heading")
        col_names.append("#ofLanes")
        col_names.append("laneWidth")
        return col_names

    def calculated_stages(self, time_list):
        """
        calculates the stages (time points where an virtual action is taken in the trajectory)

        Arguments:
            time_list [float] - list of time steps

        Return;
            [int] - list of stages for each time step
        """
        stage_list = []
        stage = 0
        for time_stamp in time_list:
            if time_stamp > float(stage + 1) * self.stage_time:
                stage += 1
            stage_list.append(stage)
        assert stage >= (self.number_of_stages + 1)
        return stage_list

    def extract_constants(self, agent_index):
        """
        Extracts the constants concerning one agent like the vehicle length

        Arguments:
            agent_index int - index of the agent for which constants should be extracted

        Return:
            tuple - tuple with vehicle length and width
        """
        df_agent_old = self.agents_dataframe[
            self.agents_dataframe["track_id"] == agent_index
        ]
        vehicle_length = df_agent_old["length"].to_numpy()[0]
        vehicle_width = df_agent_old["width"].to_numpy()[0]
        return vehicle_length, vehicle_width

    def lane_center_y_coordinate(self, lane_index):
        """
        Returns the center y coordinate of a lane

        Arguments:
            lane_index int - index of the lane

        Return:
            float - frenet y coordinate of the lane
        """
        return (float(lane_index) + 0.5) * self.lane_width

    def calculate_lane(self, frenet_y_coordinates):
        """
        Calculates the lane array corresponding to a y coordinate trajectory from a given agent

        Arguments:
            frenet_y_coordinates [float] - list of the frenet y coordinates of the trajectory

        Return:
            np.arry - array of lane indexes over time
        """
        lane_indexes = []
        for y_coordinate in frenet_y_coordinates:
            for lane_index in range(self.number_of_lanes):
                if (
                    y_coordinate >= float(lane_index) * self.lane_width
                    and y_coordinate < float(lane_index + 1) * self.lane_width
                ):
                    lane_indexes.append(lane_index)
        return np.array(lane_indexes)

    def filter_time_stamp(self, agent_dfs):
        """
        Extracts the common timesteps of all agents (agents appear in different time slots in the interaction data set)

        Arguments:
            agent_dfs pd.DataFrame - data frame from the .csv from interaction dataset

        Return:
            pd.DataFrame - filtered data frame only containing the time steps where all agents are present
        """
        time_sets = [set(df["time"].to_numpy()) for df in agent_dfs]
        current_time_set = time_sets[0]
        for time_set in time_sets[1:]:
            current_time_set = current_time_set.intersection(time_set)
        time_intersect = list(current_time_set)
        time_intersect.sort()
        agent_dfs = [df[df["time"].isin(time_intersect)] for df in agent_dfs]
        minimal_time_stamp = np.min(agent_dfs[0]["time"].to_numpy())
        for df in agent_dfs:
            df["time"] = df["time"] - minimal_time_stamp
        return agent_dfs

    def numeric_velocity_calculation(self, position_array, time_step):
        """
        numeric approximation of the velocity in a coordinate from the position (numeric first derivative)

        Arguments:
            position_array np.array - array of position in either x or y direction
            time_step float - time step between two positions in the array

        Return:
            np.array - array of the calculated velocities
        """
        position_array_t = np.array(position_array)[1:]
        position_array_t_1 = np.array(position_array)[:-1]
        velocities = (1.0 / time_step) * (position_array_t - position_array_t_1)
        return velocities

    def numeric_acceleration_calculation(self, position_array, time_step):
        """
        numeric approximation of the acceleration in a coordinate from the position (numeric second derivative)

        Arguments:
            position_array np.array - array of position in either x or y direction
            time_step float - time step between two positions in the array

        Return:
            np.array - array of the calculated acceleration
        """

        position_array_t_plus_1 = np.array(position_array)[2:]
        position_array_t = np.array(position_array)[1:-1]
        position_array_t_1 = np.array(position_array)[:-2]
        acceleration = (1.0 / np.power(time_step, 2.0)) * (
            position_array_t_plus_1 - 2 * position_array_t + position_array_t_1
        )
        return acceleration

    def create_trajectory_msg(self, agent_df, agent_index):
        """
        Submethod to create the pickled ros trajectory messages from the agent_df data frame

        Arguments:
            agent_df pd.DataFrame - pandas dataframe containing all information for the ros trajectory message
            agent_index int - index of the agent for which the ros trajectory should be created

        Return:
            msg.trajectory - trajectory message extracted from the interaction dataset .csv for agent with index agent_index
        """

        # number_of_stages = np.max(agent_df['stage'].to_numpy())
        trajectory_msg = Trajectory()
        agent_msg_generator = AgentMsgGenerator(agent_index)

        ##### Initial state #######
        agent_msg_generator.initializeNewMsg()
        pos_x, pos_y, vel_x, vel_y, acc_x, acc_y = self.get_state_in_row(
            agent_df, 0, agent_index
        )
        (
            desired_vel,
            diff_vel_vel_des,
            diff_des_lane_cent,
            abs_lane_diff,
        ) = self.get_state_features_in_row(agent_df, 0, agent_index)
        agent_msg_generator.add_state(pos_x, pos_y, vel_x, vel_y, acc_x, acc_y)
        agent_msg_generator.add_state_features(
            diff_vel_vel_des=diff_vel_vel_des,
            abs_lane_diff=abs_lane_diff,
            diff_des_lane_cent=diff_des_lane_cent,
            desired_vel=desired_vel,
            collided=False,
            invalidState=False,
            desired_lane=self.desired_lanes[agent_index],
        )
        agent_msg_generator.add_action_features(
            invalidAction=False, absoluteAverageAccY=0.0
        )

        agent_vec_msg = agent_msg_generator.get_agent_msg()
        trajectory_msg.initialState = agent_vec_msg
        trajectory_agent_vec_list = []

        for stage in range(0, self.number_of_stages + 1):
            agent_msg_generator.initializeNewMsg()
            row_index = agent_df[agent_df["stage"] == stage].last_valid_index()
            pos_x, pos_y, vel_x, vel_y, acc_x, acc_y = self.get_state_in_row(
                agent_df, row_index, agent_index
            )
            (
                desired_vel,
                diff_vel_vel_des,
                diff_des_lane_cent,
                abs_lane_diff,
            ) = self.get_state_features_in_row(agent_df, row_index, agent_index)
            agent_msg_generator.add_state(pos_x, pos_y, vel_x, vel_y, acc_x, acc_y)
            agent_msg_generator.add_state_features(
                diff_vel_vel_des=diff_vel_vel_des,
                abs_lane_diff=abs_lane_diff,
                diff_des_lane_cent=diff_des_lane_cent,
                desired_vel=desired_vel,
                collided=False,
                invalidState=False,
                desired_lane=self.desired_lanes[agent_index],
            )
            absoluteAverageAccY = self.calculate_accy_for_stage(
                agent_df, stage, agent_index
            )
            agent_msg_generator.add_action_features(
                invalidAction=False, absoluteAverageAccY=absoluteAverageAccY
            )
            agent_vec_msg = agent_msg_generator.get_agent_msg()
            trajectory_agent_vec_list.append(agent_vec_msg)
        trajectory_msg.trajectory = trajectory_agent_vec_list
        scenario_info = ScenarioInfo()
        scenario_info.laneWidth = self.lane_width
        scenario_info.scenarioName = self.scenario_name
        trajectory_msg.scenarioInfo = scenario_info
        return trajectory_msg

    def save_trajectory_msg(
        self, trajectory_msg, agent_index, pickle_file_path, pickle_prefix
    ):
        """
        Method for saving the trajectory message

        Arguments:
            trajectory_msg msg.trajectory - trajectory message to be saved
            agent_index int - index of agents that belongs to the trajectory message - used to create the file name
            pickle_file_path String - path to the subdir where the pickle files should be saved to
            pickle_prefix String - prefix for the file name
        """
        file_name = pickle_prefix + "_" + str(agent_index)
        f = open(os.path.join(pickle_file_path, file_name), "wb")
        pickle.dump(trajectory_msg, f)
        f.close()

    def calculate_accy_for_stage(self, agent_df, stage, agent_index):
        """
        Calculates acceleration feature for a given action (in a stage)

        Arguments:
            agent_df pd.DataFrame - data frame containing the trajectory information
            stage int - stage for which the feature should be calculated
            agent_index int - index of agent for which the feature should be calculated
        """
        y_acceleration_array = agent_df[agent_df["stage"] == stage][
            "m_accelerationY" + str(agent_index)
        ].to_numpy()
        mean_absoulte_accy = np.mean(np.absolute(y_acceleration_array))
        return mean_absoulte_accy

    def get_state_features_in_row(self, agent_df, row_index, agent_index):
        """
        Extracts feature values from agent_df in order to create the feature message (see msg folder)
        """

        desired_vel = agent_df["m_velocityDesiredX" + str(agent_index)][row_index]
        diff_vel_vel_des = np.abs(
            desired_vel - agent_df["m_velocityX" + str(agent_index)][row_index]
        )
        abs_lane_diff = np.abs(
            self.desired_lanes[agent_index]
            - agent_df["m_lane" + str(agent_index)][row_index]
        )
        diff_des_lane_cent = (
            self.lane_center_y_coordinate(
                agent_df["m_lane" + str(agent_index)][row_index]
            )
            - agent_df["m_positionY" + str(agent_index)][row_index]
        )
        print("row index: " + str(row_index))
        print("Desired vel: " + str(desired_vel))
        print("diff_vel_des: " + str(diff_vel_vel_des))
        print("diff_des_lane_cent: " + str(diff_des_lane_cent))
        print("Abs_lane_diff: " + str(abs_lane_diff))
        return desired_vel, diff_vel_vel_des, diff_des_lane_cent, abs_lane_diff

    def get_state_in_row(self, agent_df, row_index, agent_index):
        """
        Extracts state values from agent_df in order to create state message (see msg folder)
        """
        pos_x = agent_df["m_positionX" + str(agent_index)][row_index]
        pos_y = agent_df["m_positionY" + str(agent_index)][row_index]
        vel_x = agent_df["m_velocityX" + str(agent_index)][row_index]
        vel_y = agent_df["m_velocityY" + str(agent_index)][row_index]
        acc_x = agent_df["m_accelerationX" + str(agent_index)][row_index]
        acc_y = agent_df["m_accelerationY" + str(agent_index)][row_index]
        return pos_x, pos_y, vel_x, vel_y, acc_x, acc_y


if __name__ == "__main__":

    main_scenario_path = (
        os.environ["HOME"]
        + "/no_backup/irl_workdir/experts/realWorldExperts/scenario_2"
    )
    output_csv_file_path = os.path.join(main_scenario_path, "csv_files/")
    output_pickle_file_path = os.path.join(main_scenario_path, "pickle_files/")
    input_csv_file_path = os.path.join(main_scenario_path, "input")
    reference_file = os.path.join(main_scenario_path, "reference.osm")
    input_csv_file_names = [
        file_name
        for file_name in os.listdir(input_csv_file_path)
        if file_name.endswith(".csv")
    ]
    for input_csv_file_name in input_csv_file_names:
        try:
            print("Process " + input_csv_file_name)
            base_name = input_csv_file_name.split(".")[0]
            complete_output_csv_file_path = os.path.join(
                output_csv_file_path, base_name + "/"
            )
            os.makedirs(complete_output_csv_file_path)
            preprocessor = Preprocessor(
                number_of_lanes=3,
                lane_width=3.5,
                number_of_agents=2,
                automated_setting_of_desired_vel=True,
                automated_setting_of_desired_lane=False,
            )
            preprocessor.set_desired_lanes([1, 1])
            preprocessor.load_reference_lane(reference_file)
            preprocessor.load_input_csv(
                os.path.join(input_csv_file_path, input_csv_file_name)
            )
            base_y = 10.3
            delta_y = 0.26
            heading = -0.08
            width = 0.6
            preprocessor.add_obstacle(
                frenet_x=95.0, frenet_y=base_y, length=3.0, width=width, heading=heading
            )
            preprocessor.add_obstacle(
                frenet_x=98.0,
                frenet_y=base_y - 1 * delta_y,
                length=3.0,
                width=width,
                heading=heading,
            )
            preprocessor.add_obstacle(
                frenet_x=101.0,
                frenet_y=base_y - 2 * delta_y,
                length=3.0,
                width=width,
                heading=heading,
            )
            preprocessor.add_obstacle(
                frenet_x=104.0,
                frenet_y=base_y - 3 * delta_y,
                length=3.0,
                width=width,
                heading=heading,
            )
            preprocessor.add_obstacle(
                frenet_x=107.0,
                frenet_y=base_y - 4 * delta_y,
                length=3.0,
                width=width,
                heading=heading,
            )
            preprocessor.add_obstacle(
                frenet_x=110.0,
                frenet_y=base_y - 5 * delta_y,
                length=3.0,
                width=width,
                heading=heading,
            )
            preprocessor.add_obstacle(
                frenet_x=113.0, frenet_y=8.9, length=5.0, width=1.0, heading=0.0
            )
            preprocessor.add_obstacle(
                frenet_x=118.0, frenet_y=8.9, length=5.0, width=1.0, heading=0.0
            )
            preprocessor.add_obstacle(
                frenet_x=123.0, frenet_y=8.9, length=5.0, width=1.0, heading=0.0
            )
            preprocessor.add_obstacle(
                frenet_x=128.0, frenet_y=8.9, length=5.0, width=1.0, heading=0.0
            )
            preprocessor.add_obstacle(
                frenet_x=133.0, frenet_y=8.9, length=5.0, width=1.0, heading=0.0
            )
            preprocessor.add_obstacle(
                frenet_x=138.0, frenet_y=8.9, length=5.0, width=1.0, heading=0.0
            )
            preprocessor.create_output_files(
                output_csv_file_path=complete_output_csv_file_path,
                output_pickle_file_path=output_pickle_file_path,
                pickle_file_prefix=base_name,
            )
        except:
            print("Couldn't process " + input_csv_file_name)
