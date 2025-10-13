from envs.grocery_details import DIMS
from envs import __file__ as base_path
import numpy as np
import os.path


def gen_drawer_xml(dim, ypos=0, num_shelves=3):
    """
    Generates an mjcf xml description of a table.

    Args:
    pos: specifies the position of the table in global coordinates
    dim: specifies the length, width and height of the table
    num_shelves: number of shelves to add. 
    """
    # generate frame
    length, width, height = map(lambda x: x / 2, dim)
    THICKNESS = 0.01  # wall thickness
    POS_THICKNESS = 0.01
    drawer_size = (height) / num_shelves
    xml = f"""
        <body name="cabinet" pos="0 {ypos:.3f} 0">
            <!-- Cabinet: Hollow Frame (No Front) -->
            <!-- Bottom of the Cabinet Frame -->
            <geom name="frame_bottom" type="box" size="{length + THICKNESS:.3f} {width + THICKNESS:.3f} {THICKNESS:.3f}" pos="0 0 0" material="wood"/>
            <!-- Top of the Cabinet Frame -->
            <geom name="frame_top" type="box" size="{length + THICKNESS:.3f} {width + THICKNESS:.3f} {THICKNESS:.3f}" pos="0 0 {height * 2:.3f}" material="wood" />
            <!-- Left Side of the Cabinet Frame -->
            <geom name="frame_left" type="box" size="{THICKNESS:.3f} {width + THICKNESS:.3f} {height:.3f}" pos="-{length:.3f} 0 {height:.3f}" material="wood" />
            <!-- Right Side of the Cabinet Frame -->
            <geom name="frame_right" type="box" size="{THICKNESS:.3f} {width + THICKNESS:.3f} {height:.3f}" pos="{length:.3f} 0 {height:.3f}" material="wood" />
            <!-- Back of the Cabinet Frame -->
            <geom name="frame_back" type="box" size="{length + THICKNESS:.3f} {THICKNESS:.3f} {height:.3f}" pos="0 {width:.3f} {height:.3f}" material="wood" />
    """
    for i in range(num_shelves):
        xml += f"""
            <body name="drawer{i}" pos="0 0 {(1 + 2*i)*drawer_size:.3f}">
                <!-- Bottom of Drawer 1 -->
                <geom name="drawer{i}_bottom" type="box" size="{length - THICKNESS:.3f} {width - THICKNESS:.3f} {THICKNESS:.3f}" pos="0 -{2*POS_THICKNESS:.3f} -{drawer_size - 2*POS_THICKNESS:.3f}" material="mat_drawer" contype="2" />
                <geom name="d{i}_left" type="box" size="{THICKNESS:.3f} {width - THICKNESS:.3f} {drawer_size - THICKNESS:.3f}" pos="-{length - 2*POS_THICKNESS:.3f} -{2*POS_THICKNESS:.3f} 0" material="mat_drawer" contype="4"/>
                <geom name="d{i}_right" type="box" size="{THICKNESS:.3f} {width - THICKNESS:.3f} {drawer_size - THICKNESS:.3f}" pos="{length - 2*POS_THICKNESS:.3f} -{2*POS_THICKNESS:.3f} 0" material="mat_drawer" contype="4" />
                <geom name="d{i}_front" type="box" size="{length - THICKNESS:.3f} {THICKNESS:.3f} {drawer_size - THICKNESS:.3f}" pos="0 -{width:.3f} 0" material="mat_drawer" contype="8"/>

                <geom name="handle{i}_left" material="slide_metal" quat="1 -1 0 0"  size="0.005 0.0155 0.005" pos="-0.05 -{width + .015:.3f} 0" type="cylinder"/>
                <geom name="handle{i}_right" material="slide_metal" quat="1 -1 0 0"  size="0.005 0.0155 0.005" pos="0.05 -{width + .015:.3f} 0" type="cylinder"/>
                <geom name="handle{i}" material="slide_metal" quat="1 0 -1 0" size="0.005 0.05 0.005" pos="0 -{width + .03:.3f}  0" type="cylinder"/>
                <joint name="drawer{i}_slide" type="slide" axis="0 -1 0" pos="0 {width:.3f} 0" range="0 {width:.3f}" damping="0" stiffness="0"/>
            </body>
        """
    xml +="</body>"
    return xml

def gen_arrangement(
    drawer_y_pos=0.9,
    drawer_size=[0.6, 0.4, 0.7],
    num_shelves=3,
    add_object=False,
) -> str:
    """
    Generates xml for an arrangement and places all the grocery items out of view.

    Arguments:
    arrangement_type: FRONTAL_GRASP = 0, PARTIAL_OCCLUSION = 1, FULL_OCCLUSION = 2
    """
    drawer_pos = np.array([0, drawer_y_pos, 0])
    drawer_size = np.array(drawer_size)

    location = os.path.dirname(os.path.realpath(base_path))
    base_scene = os.path.join(location, "scenes", "base_scene.xml")
    with open(base_scene, "r") as f:
        xml = "\n".join(f.readlines())
    xml += gen_drawer_xml(drawer_size, ypos = drawer_y_pos, num_shelves=num_shelves)
    if add_object:
        items = list(DIMS.keys())
        item = np.random.choice(items)
        cnt = 0
        for cnt, item in enumerate(items):
            item_name = item.split("_")[0]
            xml += f"""<body name="{item}" pos="{100 + cnt} {0} {0}" quat="1 1 0 0">
                        <freejoint name="{item}"/>
                            <geom material="{item_name}" mesh="{item_name}" class="visual" shellinertia="true" mass="{0.5}"/>
                            <geom name="{item}" mesh="{item_name}" class="collision"/>
                        </body>"""
    xml += f"""
    </worldbody>
    </mujoco>
    """
    a = "_".join(map(lambda x: str(int(round(x, 2) * 1000)), list(drawer_size)))
    # fname = os.path.join(location, f"scenes/scene_{a}_{np.random.randint(2**32)}.xml")
    # with open(fname, "w") as f:
    #     f.write(xml)
    fname = os.path.join(location, "scenes/scene_600_400_700_1902111393.xml")
    # fname = os.path.join(location, "scenes/scene_600_280_590_1553682431.xml")
    return fname, {"drawer_pos": drawer_pos, "drawer_size": drawer_size}


if __name__ == "__main__":
    print(gen_arrangement())