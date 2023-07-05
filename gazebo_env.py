# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:14:18 2022

@author: Administrator
"""
import torch
import rospkg
import rospy
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan

import math
import numpy as np
import time
import sys
import csv
CONSTANTS_RADIUS_OF_EARTH = 6371000.     # meters (m)

#到达目标点判定范围
# dis = 5.0
# max of laserscan
MAXLASERDIS = 15
#奖励函数参数
bad = -10000
e = -2.0
f = 5

class Env_model():
    def __init__(self):
        rospy.init_node('control_node',anonymous = True)

        self.agentrobot = 'rexrov'
        self.dis = 10.0   #位置精度
        self.obs_pos = []  # 障碍物的位置信息
        self.filename = '|home|elon|data|uuv.csv'
        self.sensor_position = np.zeros(3)
        self.sensor_rpy = np.zeros(3)
        self.sensor_quaternion = np.zeros(4)

        self.gazebo_model_states = ModelStates()#机器人位姿讯息
        self.resetval()#初始化各参数

        # 接收gazebo的modelstate消息
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_states_callback)

        # 接收agent robot的声呐信息
        self.subLaser = rospy.Subscriber('/rexrov/sonar', LaserScan, self.sonar_states_callback)
        # 发布控制指令给agent robot
        self.pub = rospy.Publisher('/rexrov/cmd_vel', Twist, queue_size=10)
        #接收robot的angel
        self.odometry_sub = rospy.Subscriber('/rexrov/pose_gt',Odometry,self.sensor_state_callback)

        time.sleep(1.0)

    def resetval(self):
        self.robotstate = [0.0, 0.0, 0.0, 0.0]  # x,y,v,w
        self.d          = 0.0                                  # 到目标的距离
        self.d_last     = 0.0                                  # 前一时刻到目标的距离
        self.v_last     = 0.0                                  # 前一时刻的速度
        self.w_last     = 0.0                                  # 前一时刻的角速度
        self.r          = 0.0                                  # 奖励
        self.cmd        = [1.0, 0.0]                           # agent robot的控制指令:v,w
        self.done_list  = False                                # episode是否结束的标志



    #接收机器人位姿信息
    def gazebo_states_callback(self, data):
        self.gazebo_model_states = data




    #接收激光雷达信息
    def sonar_states_callback(self, data):
        self.sonar = data
        self.range_min = data.range_min

    #地图角度转换
    def angel(self,point_A,point_B,point_C):
        a = math.sqrt((point_B[0]-point_C[0])**2+(point_B[1]-point_C[1])**2)
        b = math.sqrt((point_A[0]-point_C[0])**2+(point_A[1]-point_C[1])**2)
        c = math.sqrt((point_A[0]-point_B[0])**2+(point_A[1]-point_B[1])**2)
        C = (c**2+a**2-b**2)/(2*a*c)
        return C

    def sensor_state_callback(self,msg):
        if not isinstance(msg,Odometry):
            return
        self.robotstate[0] = msg.pose.pose.position.x
        self.robotstate[1] = msg.pose.pose.position.y
        v = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        self.robotstate[2] = v
        self.robotstate[3] = msg.twist.twist.angular.z
        self.sensor_position[0] = msg.pose.pose.position.x
        self.sensor_position[1] = msg.pose.pose.position.y
        self.sensor_position[2] = msg.pose.pose.position.z
        self.sensor_quaternion[0] = msg.pose.pose.orientation.x
        self.sensor_quaternion[1] = msg.pose.pose.orientation.y
        self.sensor_quaternion[2] = msg.pose.pose.orientation.z
        self.sensor_quaternion[3] = msg.pose.pose.orientation.w

        self.quaternion_to_rpy


    @property
    def quaternion_to_rpy(self):
        """`numpy.array`: Orientation error in Euler angles."""
        q_1 = self.sensor_quaternion[0]
        q_2 = self.sensor_quaternion[1]
        q_3 = self.sensor_quaternion[2]
        q_0 = self.sensor_quaternion[3]


        g_1 = 2*(q_3*q_2-q_0*q_1)
        g_2 = q_0**2-q_1**2-q_2**2+q_3**2
        g_3 = 2*(q_0*q_2-q_1*q_3)
        g_4 = 2*(q_1*q_2+q_0*q_3)
        g_5 = q_0**2+q_1**2-q_2**2-q_3**2
        roll = (np.arctan2(g_1,g_2))*180/np.pi
        pitch = (np.arcsin(g_3))*180/np.pi
        yaw = (np.arctan2(g_4,g_5))*180/np.pi
        self.sensor_rpy = np.array([roll,pitch,yaw])
        return self.sensor_rpy



    def getreward(self):

        R1 = 500.0 * (self.d_last - self.d)#go to the position
        R2 = -1.0 * abs(self.w_last - 0)#w恒定
        
        #快速到达
        R3 = -3.0
        #touch the obstacle
        if self.range_min < 2:
            R4 = -10000
        #get away from the obstacle

        if self.d < self.dis:
            R5 = 10000
        else:
            R5 = 0

        R6 = 10.0 * (self.angel([0.0,0.0], [50.0,50.0], [self.robotstate[0],self.robotstate[1]]) - 1.0)

        r = R1 + R2 + R3 + R4 + R5 + R6

        return r

    def step(self, action = [0,0]):
        self.d_last = math.sqrt((self.robotstate[0] - self.gp[0])**2 + (self.robotstate[1] - self.gp[1])**2)
        self.cmd[0] = action[0]
        self.cmd[1] = action[1]
        cmd_vel = Twist()
        cmd_vel.linear.x  = self.cmd[0]
        cmd_vel.angular.z = action[0]
        self.pub.publish(cmd_vel)

        time.sleep(0.05)

        self.d = math.sqrt((self.robotstate[0] - self.gp[0])**2 + (self.robotstate[1] - self.gp[1])**2)
        _xy = [self.robotstate[0],self.robotstate[1]]
        self.robotxy.append(_xy)


        self.v_last = action[0]
        self.w_last = action[1]


        self.next_state = []
        self.next_state.append(self.sensor_rpy[2])
        self.next_state.append(self.d)
        self.next_state.append(self.v_last)
        self.next_state.append(self.w_last)
        self.next_state.append(self.temp(self.sonar))
        self.next_state = self.normalazition(self.next_state)


        self.r = self.getreward()

        # 是否到达目标点判断
        if self.d > self.dis:
            self.done_list = False  # 不终止
        else:
            self.done_list = True  # 终止
            print("Goal Point!")
            self.data_write_csv(self.filename, self.robotxy)
        if self.range_min < 2:
            self.done_list = True
            print("Get the tackle!")


        return self.next_state, self.r, self.done_list

    def temp(self):
        sonar = []
        temp = []
        # sensor_info = []
        for j in range(len(self.laser.ranges)):
            tempval = self.laser.ranges[j]
            # 归一化处理
            if tempval > MAXLASERDIS:
                tempval = MAXLASERDIS
            temp.append(tempval/MAXLASERDIS)
        sonar = temp
        return sonar


    def normalazition(self,data):
        _max = np.max(data)
        _min = np.min(data)
        if abs(_max)>abs(_min):
            temp = abs(_max)
        else:
            temp = abs(_min)
        return data / temp

    def data_write_csv(self,file_name,datas):
        file_csv = open(file_name,'w')
        writer = csv.writer(file_csv)
        for data in datas:
            writer.writerow(data)


    def quaternion_from_euler(self, r, p, y):
        q = [0, 0, 0, 0]
        q[3] = math.cos(r / 2) * math.cos(p / 2) * math.cos(y / 2) + math.sin(r / 2) * math.sin(p / 2) * math.sin(y / 2)
        q[0] = math.sin(r / 2) * math.cos(p / 2) * math.cos(y / 2) - math.cos(r / 2) * math.sin(p / 2) * math.sin(y / 2)
        q[1] = math.cos(r / 2) * math.sin(p / 2) * math.cos(y / 2) + math.sin(r / 2) * math.cos(p / 2) * math.sin(y / 2)
        q[2] = math.cos(r / 2) * math.cos(p / 2) * math.sin(y / 2) - math.sin(r / 2) * math.sin(p / 2) * math.cos(y / 2)
        return q

    def reset(self, start=[0.0,0.0], goal=[50.0, 50.0]):
        self.sp = start
        self.gp = goal
        self.robotxy = []
        self.ob1 = [30,80]
        self.ob2 = [-70,-70]

        # 初始点到目标点的距离
        self.d_sg = ((self.sp[0]-self.gp[0])**2 + (self.sp[1]-self.gp[1])**2)**0.5

        # 重新初始化各参数
        self.resetval()
        rospy.wait_for_service('/gazebo/set_model_state')
        val = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)


        # agent robot生成一个随机的角度
        # 根据model name对每个物体的位置初始化
        state = ModelState()
        for i in range(len(self.gazebo_model_states.name)):
            if self.gazebo_model_states.name[i] == "point_start":
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                state.pose.position.x = self.sp[0]
                state.pose.position.y = self.sp[1]
                val(state)
            if self.gazebo_model_states.name[i] == "ob1":
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                state.pose.position.x = self.ob1[0]
                state.pose.position.y = self.ob1[1]
                val(state)
            if self.gazebo_model_states.name[i] == "ob2":
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                state.pose.position.x = self.ob2[0]
                state.pose.position.y = self.ob2[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot:
                state.reference_frame = 'map'
                state.pose.position.z = -10.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, 45.0]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = self.sp[0]
                state.pose.position.y = self.sp[1]
                val(state)
                
        self.done_list = False  # episode结束的标志
        model_state = []

        model_state.append(self.sensor_rpy[2])
        model_state.append(self.d)
        model_state.append(self.v_last)
        model_state.append(self.w_last)
        model_state.append(self.temp(self.sonar))

        return self.normalazition(model_state)
        print("The environment has been reset!")

        time.sleep(2.0)


if __name__ == '__main__':
    pass
