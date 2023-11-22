import numpy as np

from pinocchio import RobotWrapper, SE3, Model, Inertia, JointModelSpherical, JointModelRUBY, GeometryModel, GeometryObject, Frame, FrameType
from pinocchio.utils import rotate
from hppfcl import Cylinder, Sphere

# Constants
ARM_SEGMENT_MASS = 1.0
UPPER_ARM_RADIUS = 0.05
FOREAMR_RADIUS = 0.04
JOINT_SPHERE_RADIUS_FACTOR = 1.1

SHOULDER_INITIAL_Z = 1.0
SHOULDER_INITIAL_ROTATION = rotate('x', np.pi / 2.0)

JOINT_SPHERE_COLOR = np.array([0.8, 0.3, 0.3, 1.0])
UPPER_ARM_COLOR = np.array([0.3, 0.8, 0.3, 0.6])
FOREARM_COLOR = np.array([0.3, 0.3, 0.8, 8.0])

END_EFFECTOR_FRAME_RADIUS = 0.01
END_EFFECTOR_FRAME_LENGTH = 0.1



# Arm definition
class ArmRobot(RobotWrapper):

    def __init__(self, upper_arm_length:float, forearm_length: float, headless: bool=False, verbose: bool=False):

        # Initialize the arm model
        model = self._build_model(upper_arm_length, forearm_length)

        # Initialize the arm collision model 
        collision_model = GeometryModel() # Useless apriori, but cannot be None (else error with viz)

        # Initialize the arm visual model
        visual_model = None
        if not headless:
            visual_model = self._build_visual_model(upper_arm_length, forearm_length, model)

        super().__init__(model, collision_model, visual_model, verbose)

    def _build_model(self, upper_arm_length: float, forearm_length: float):
        model = Model()

        # Inertia for the arm part
        upper_arm_inertia = Inertia.FromCylinder(ARM_SEGMENT_MASS, UPPER_ARM_RADIUS, upper_arm_length)
        forearm_inertia = Inertia.FromCylinder(ARM_SEGMENT_MASS, FOREAMR_RADIUS, forearm_length)

        # Add shoulder joint (spherical) to model
        shoulder_joint = JointModelSpherical()
        shoulder_joint_id = model.addJoint(
            0, 
            shoulder_joint,
            SE3(
                SHOULDER_INITIAL_ROTATION,
                np.array([0.0, 0.0, SHOULDER_INITIAL_Z])
            ),
            "shoulder",
        )

        # Add inertia to the shoulder joint
        model.appendBodyToJoint(
            shoulder_joint_id,
            upper_arm_inertia,
            SE3(
                np.eye(3),
                np.array([0.0, 0.0, upper_arm_length / 2.0])
            ),
        )

        # Add elbow joint (revolute) to model
        elbow_joint = JointModelRUBY()
        elbow_joint_id = model.addJoint(
            shoulder_joint_id,
            elbow_joint,
            SE3(
                np.eye(3),
                np.array([0.0, 0.0, upper_arm_length]),
            ),
            "elbow",
        )

        # Add inertia to the elbow joint
        model.appendBodyToJoint(
            elbow_joint_id,
            forearm_inertia,
            SE3(
                np.eye(3), 
                np.array([0.0, 0.0, forearm_length / 2.0]),
            ),
        )
        
        # Add end-effector frame at the end of the forearm
        end_effector_frame = Frame(
            "end_effector", 
            elbow_joint_id, 
            0,
            SE3(
                np.eye(3), 
                np.array([0.0, 0.0, forearm_length])
            ),
            FrameType.OP_FRAME,
        )
        model.addFrame(end_effector_frame)

        return model


    def _build_visual_model(self, upper_arm_length: float, forearm_length: float, model: Model):
        visual_model = GeometryModel()

        # GeometryObject for the shoulder joint
        shoulder_geometry = GeometryObject(
            "shoulder_geom", 
            0,
            model.getJointId("shoulder"),  
            Sphere(JOINT_SPHERE_RADIUS_FACTOR * UPPER_ARM_RADIUS),  
            SE3(
                np.eye(3), 
                np.zeros(3)
            ), 
        )
        shoulder_geometry.meshColor = JOINT_SPHERE_COLOR

        # GeometryObject for the upper_arm
        upper_arm_geometry = GeometryObject(
            "upper_arm_geom", 
            0,
            model.getJointId("shoulder"),  
            Cylinder(UPPER_ARM_RADIUS, upper_arm_length),  
            SE3(
                np.eye(3), 
                np.array([0.0, 0.0, upper_arm_length / 2.0])
            ), 
        )
        upper_arm_geometry.meshColor = UPPER_ARM_COLOR

        # GeometryObject for the elbow joint
        elbow_geometry = GeometryObject(
            "elbow_geom", 
            0,
            model.getJointId("elbow"),  
            Sphere(JOINT_SPHERE_RADIUS_FACTOR * UPPER_ARM_RADIUS),  
            SE3(
                np.eye(3), 
                np.zeros(3)
            ), 
        )
        elbow_geometry.meshColor = JOINT_SPHERE_COLOR

        # GeometryObject for the upper_arm
        forearm_geometry = GeometryObject(
            "forearm_geom", 
            0,
            model.getJointId("elbow"),  
            Cylinder(FOREAMR_RADIUS, forearm_length),  
            SE3(
                np.eye(3), 
                np.array([0.0, 0.0, forearm_length / 2.0])
            ),
        )
        forearm_geometry.meshColor = FOREARM_COLOR
        
        # Geometry objects for the end-effector frame
        end_effector_frame_id = model.getFrameId("end_effector")
        parent_frame_id = model.frames[end_effector_frame_id].parent

        x = rotate('y', np.pi / 2.0)
        y = rotate('x', -np.pi / 2.0)
        z = np.eye(3)
        position_effector = np.array([0.0, 0.0, forearm_length])
        med = np.array([0.0, 0.0, END_EFFECTOR_FRAME_LENGTH / 2.0])

        frame_x_geometry = GeometryObject(
            "frame_axis_x", 
            end_effector_frame_id, 
            parent_frame_id, 
            Cylinder(END_EFFECTOR_FRAME_RADIUS, END_EFFECTOR_FRAME_LENGTH),
            SE3(
                x,
                x @ med + position_effector
            ),
        )
        frame_x_geometry.meshColor = np.array([1.0, 0.0, 0.0, 1.0])

        frame_y_geometry = GeometryObject(
            "frame_axis_y", 
            end_effector_frame_id, 
            parent_frame_id, 
            Cylinder(END_EFFECTOR_FRAME_RADIUS, END_EFFECTOR_FRAME_LENGTH),
            SE3(
                y,
                y @ med + position_effector
            ),
        )
        frame_y_geometry.meshColor = np.array([0.0, 1.0, 0.0, 1.0])

        frame_z_geometry = GeometryObject(
            "frame_axis_z", 
            end_effector_frame_id, 
            parent_frame_id, 
            Cylinder(END_EFFECTOR_FRAME_RADIUS, END_EFFECTOR_FRAME_LENGTH),
            SE3(
                z,
                z @ med + position_effector
            ),
        )
        frame_z_geometry.meshColor = np.array([0.0, 0.0, 1.0, 1.0])

        # Append geometric objects to the geometry model
        visual_model.addGeometryObject(shoulder_geometry)
        visual_model.addGeometryObject(upper_arm_geometry)
        visual_model.addGeometryObject(elbow_geometry)
        visual_model.addGeometryObject(forearm_geometry)
        visual_model.addGeometryObject(frame_x_geometry)
        visual_model.addGeometryObject(frame_y_geometry)
        visual_model.addGeometryObject(frame_z_geometry)

        return visual_model

