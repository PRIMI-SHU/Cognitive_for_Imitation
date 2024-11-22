#!/usr/bin/env python3
####Real robot experiment, should put it in the ROS package for controlling
import rospy,os,sys
import torch
import torch.nn as nn

import cv2
#package path
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
module_dir = os.path.join(main_dir)

sys.path.append(module_dir)



from model.util import helper
from model.networks import metric,vae_blend
from model.networks.dynamic_planner  import Dynamic_Planner
from model.mvae import Agent

from std_msgs.msg import String,Float64MultiArray,MultiArrayDimension
import numpy as np
import math
import moveit_commander
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image,JointState
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction,FollowJointTrajectoryActionGoal,FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
import kdl_parser_py.urdf as kdl_parser
from PyKDL import ChainFkSolverPos_recursive, JntArray,Frame
import threading
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config=helper.Hparams()

dataset=helper.MVAE_DATA(config.babbling_data,config.blend_ann,'train')
def noise(x):
    x_noise= - 2 * torch.ones_like(x,dtype=torch.float)
    return x_noise

class MVAE_Dynamic():
    def __init__(self,arm,model):
        self.model=model
        
        self.arm=arm
        self.bridge = CvBridge()
        self.robot_config=None
        self.s_v=torch.tensor(np.float32(np.zeros((1, 1, 128, 128))),
                                device=device, dtype=torch.float, requires_grad=True)
        self.image=None
        self.joint_tensor=torch.zeros((1,5),dtype=torch.float).to(device)
        self.tip_tensor=torch.zeros((1,3),dtype=torch.float).to(device)
        self.goal=None
        self.tip=None
        
        self.color_pub = rospy.Publisher('robot_result', Image, queue_size=10)
        urdf_param= '/robot_description'
        (ok, tree) = kdl_parser.treeFromParam(urdf_param)
        base_link = "base"
        end_effector_link = "right_l6"
        self.chain = tree.getChain(base_link, end_effector_link)
        self.fk_solver = ChainFkSolverPos_recursive(self.chain)
        self.num_joints=self.chain.getNrOfJoints()
        self.bridge=CvBridge()

        
    
    def fwd_kinematic(self,joints):
        joint_array=JntArray(self.num_joints)
        for i,value in enumerate(joints):
            joint_array[i]=value
        end_effector_frame = Frame()
        self.fk_solver.JntToCart(joint_array, end_effector_frame)
        
        end_effector_pose =[]
        for i in range(3):
            end_effector_pose.append(end_effector_frame.p[i])
    
        return end_effector_pose
    
        
    def reset(self,joint_start):
        joint=dataset.de_normalize(joint_start.detach().cpu().numpy(),dataset.offset_joint,dataset.scale_joint)
        
        pos1=np.array([0.4632041015625, 0.88615625, -2.139068359375, 1.247962890625, 1.0991123046875, 0.44265234375, 0.0848232421875])
        pos1[0],pos1[2],pos1[3],pos1[4],pos1[5]=joint[0],joint[1],joint[2],joint[3],joint[4]
       
        self.arm.set_joint_value_target(pos1)
        self.arm.go()
        return joint
        
    def image_callback(self,msg):
        try:
            image=msg
            img = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            
            
            self.image=img
            
            x_start,x_end=0,640
            y_start,y_end=110,480
            img=img[y_start:y_end, x_start:x_end]
            
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            img=cv2.resize(img,(128,128))
            init_image=img.astype("float32")/255
            init_image=torch.tensor(init_image,dtype=torch.float)
            init_image=torch.reshape(init_image,(1,1,128,128))        
            self.s_v=init_image 
            
        except CvBridge as e:
            print(e)
            
    def save_image(self,index):
        
        init_image = self.s_v.squeeze().detach().cpu().numpy()

        image_numpy = (init_image * 255).astype(np.uint8)
        
        cv2.imwrite(f"{index}.jpg", image_numpy)
        
            
    def display(self,box,goal):
        
        if self.image is not None:
            background_img=self.image
            box=np.array(box,dtype=np.int32)

            box=box.reshape((4,2))
            
            
            center_x = np.mean(box[:, 0]).astype(int)
            center_y = np.mean(box[:, 1]).astype(int)

            #
            img1=goal*255
            im1=img1.detach().cpu().numpy()
            im1=im1.transpose(1, 2, 0)
            cv2.circle(background_img, (center_x, center_y), 5, (0, 0, 255), -1)
            x_start,x_end=0,640
            y_start,y_end=110,480
            background_img=background_img[y_start:y_end, x_start:x_end]

            scale_factor = 1.3

            new_width = int(im1.shape[1] * scale_factor)
            new_height = int(im1.shape[0] * scale_factor)

            
            resized_overlay_img = cv2.resize(im1, (new_width, new_height))
            resized_overlay_img = cv2.cvtColor(resized_overlay_img, cv2.COLOR_GRAY2BGR)
         
            color_msg = self.bridge.cv2_to_imgmsg(background_img, encoding="bgr8")
            color_msg.header.stamp=rospy.Time.now()
            self.color_pub.publish(color_msg)
            return background_img
            
            
            
                        
        
    def set_goal(self,image):
        
        noise_joint=noise(self.joint_tensor)
        noise_tip=noise(self.tip_tensor)
        _,out_mu,_=self.model.map_recon([image,noise_joint.to(device),noise_tip.to(device)])
        
        joint=out_mu[1][0] 
        tip_de=dataset.de_normalize(out_mu[2][0].detach().cpu().numpy(),dataset.offset_tip,dataset.scale_tip)    
        self.goal=tip_de
        return joint
    
    def self_robot(self):
        
        if self.s_v!=None:
            noise_tip=noise(self.tip_tensor)
            noise_joint=noise(self.joint_tensor)
            _,out_mu,_=self.model.self_recon([self.s_v.to(device),noise_joint.to(device),noise_tip.to(device)])
            tip_de=dataset.de_normalize(out_mu[2][0].detach().cpu().numpy(),dataset.offset_tip,dataset.scale_tip) 
            self.tip=tip_de
            joint_de=dataset.de_normalize(out_mu[1][0].detach().cpu().numpy(),dataset.offset_joint,dataset.scale_joint)
           
            return tip_de,joint_de
    
    def actions(self,diff):
        diff=torch.tensor(diff).to(device)
        action=self.model.action(diff.float())
        action= action.cpu().detach().numpy()
        action=action.reshape(-1)
        
        return action
    
    def get_joints(self):
        joint=[]
        while len(joint)<9:
            joints=rospy.wait_for_message('/robot/joint_states',JointState)
            joint=joints.position
            
        self.joints=[joint[1],joint[2],joint[3],joint[4],joint[5],joint[6]]
        pose=self.fwd_kinematic(self.joints)
        return self.joints,pose
        
    
    def move(self,action):
        joints,_=self.get_joints()
        
        joints[0],joints[2],joints[3],joints[4],joints[5]=joints[0]+action[0],joints[2]+action[1],joints[3]+action[2],joints[4]+action[3],joints[5]+action[4]
        joints=list(joints)
        joints.append(0.0842041015625)
        self.arm.set_joint_value_target(joints)
        plan = self.arm.plan()
        self.arm.go(wait=True) 

def publish_image(agent,rate):
        global exit_flag
        while not rospy.is_shutdown() and not exit_flag:
            if shared_data['box'] is not None:
                agent.display(shared_data['box'],shared_data['goal'])
            rate.sleep()
if __name__=='__main__':
    
    shared_data = {
    'box': None,
    'goal': None
    }
    
    exit_flag=False    
    nh=rospy.init_node('robot',anonymous=True)
    
       
    
    moveit_commander.roscpp_initialize(sys.argv)
    joint_state_topic = ['joint_states:=/robot/joint_states']
    moveit_commander.roscpp_initialize(joint_state_topic)	
    arm=moveit_commander.MoveGroupCommander('right_arm')
    reference_frame = 'base'
    arm.set_pose_reference_frame(reference_frame)
    arm.set_goal_joint_tolerance(0.001)
    arm.set_max_acceleration_scaling_factor(0.1)
    arm.set_max_velocity_scaling_factor(0.1)
    arm.set_planer_id = "RRTkConfigDefault"
    arm.set_planning_time(50)
    
    
    config=helper.Hparams()
    
    
    m_net=metric.Metric(config)
    m_net.load_state_dict(torch.load(config.metric_path))
    m_net.eval()
    vae=vae_blend.blend_vae(config.in_chanels,config.out_chanels,config.in_shared,config.out_shared)
    vae.load_state_dict(torch.load(config.mvae_path))
    vae.eval()   

    dynamic_model=Dynamic_Planner(3,5,3,128)
    dynamic_model.load_state_dict(torch.load(config.dynamic_path))
    dynamic_model.eval()
    
    model=Agent(vae,m_net,dynamic_model,device)
    
    agent=MVAE_Dynamic(arm,model)
    
    camera_topic = "/camera/aligned/color/image_raw"
    rospy.Subscriber(name=camera_topic,
                        data_class=Image,
                        callback=agent.image_callback, 
                        queue_size=1)
    
    c=helper.Contrastive_DATA(config.anchor_data,config.compare_data,config.full_ann,'test')
    dataset=helper.MVAE_DATA(config.babbling_data,config.blend_ann,'train')
    init=np.random.randint(0,3165)
    init_image,joint_start,tip_start,gripper_start,_,_,_,_=dataset[init]
    
    
    
    rate=rospy.Rate(10)
    
    image_thread = threading.Thread(target=publish_image, args=(agent, rate))  ##mutli threading function for displaying the ground truth 3D pint in camera space
    image_thread.start()
    
    joint1=agent.reset(joint_start)
    
    
    tip,joint=agent.self_robot()
    
    #These codes can be used to see the reconstrcuted joints on the robot:
    
    # index_goal=np.random.randint(0,len(c))
    # print(index_goal)
    # x,goal,joint,tip_,box=c[index_goal] 
    # joint=agent.set_goal(goal.unsqueeze(0).to(device))
    
    # agent.reset(joint)
    # agent.save_image(index_goal)
    
    
    try:
       
    
        for i in range(5):
            if rospy.is_shutdown() or exit_flag:
                break
            index_goal=np.random.randint(0,len(c))
            print(index_goal)
            x,goal,joint,tip_,box=c[index_goal] 
            shared_data['box']=box
            shared_data['goal']=goal
            
            agent.set_goal(goal.unsqueeze(0).to(device))
            tip_=tip_.detach().cpu().numpy()
            joint_=joint.detach().cpu().numpy()
            for _ in range(10):
                if rospy.is_shutdown() or exit_flag:
                    break
                
                tip,joint=agent.self_robot()
                
                diff=agent.goal-tip
                
                # diff1=tip_-tip
                # joint_diff=joint_-joint
                if np.mean(abs(diff))<=0.03:
                    break
                action=agent.actions(diff)
                agent.move(action)
            
            
            
                # print('joint diff:',joint_diff)
    except (rospy.ROSInterruptException,KeyboardInterrupt):
        rospy.loginfo('Stopping both threads')
        exit_flag=True
    
    finally:
        rospy.loginfo('Shutting down successfully') 
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        