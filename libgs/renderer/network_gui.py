import json
import socket
import traceback

import torch
from jaxtyping import Float
from torch import Tensor

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


class MiniCam:
    def __init__(
        self,
        width: int,
        height: int,
        fovy: float,
        fovx: float,
        znear: float,
        zfar: float,
        world_view_transform: Float[Tensor, "4 4"],
        full_proj_transform: Float[Tensor, "4 4"],
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = self.world_view_transform.inverse()
        self.camera_center = view_inv[3][:3]


def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)


def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except Exception:
        pass


def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, "little")
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))


def send(message_bytes, verify):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, "little"))
    conn.sendall(bytes(verify, "ascii"))


def receive():
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(
                torch.tensor(message["view_matrix"]), (4, 4)
            ).cuda()
            world_view_transform[:, 1] = -world_view_transform[:, 1]
            world_view_transform[:, 2] = -world_view_transform[:, 2]
            full_proj_transform = torch.reshape(
                torch.tensor(message["view_projection_matrix"]), (4, 4)
            ).cuda()
            full_proj_transform[:, 1] = -full_proj_transform[:, 1]
            custom_cam = MiniCam(
                width,
                height,
                fovy,
                fovx,
                znear,
                zfar,
                world_view_transform,
                full_proj_transform,
            )
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return (
            custom_cam,
            do_training,
            do_shs_python,
            do_rot_scale_python,
            keep_alive,
            scaling_modifier,
        )
    else:
        return None, None, None, None, None, None


def interact_with_gui(
    current_step,
    pipe,
    module,
    source_path,
    max_steps,
):
    if conn == None:
        try_connect()

    def routine():
        net_image_bytes = None
        (
            custom_cam,
            do_training,
            pipe.convert_SHs_python,
            pipe.compute_cov3D_python,
            keep_alive,
            scaling_modifer,
        ) = receive()
        if custom_cam != None:
            net_image = module(custom_cam, scaling_modifer=scaling_modifer)["render"]
            net_image_bytes = memoryview(
                net_image.clamp(0, 1)
                .mul(255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
        send(net_image_bytes, source_path)
        return do_training, keep_alive

    while conn != None:
        try:
            do_training, keep_alive = routine()
            if do_training and ((current_step < max_steps) or not keep_alive):
                return
        except Exception:
            pass
