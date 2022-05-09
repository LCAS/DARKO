from logging import exception
from constants import *
from math import sqrt, atan2
from log import log
from os import listdir
from os.path import isfile, join
from darko_causal_discovery_msgs.msg import CausalModel 
import shutil


def file_in_folder(dir_path):
    """
    List files in a specific folder

    Args:
        dir_path (str): folder path

    Returns:
        list[str]: list of files found in the folder
    """
    prefix = "data_"
    ext = ".csv"
    files_list = [f[len(prefix) : len(f) - len(ext)] for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return files_list


def create_data_dir():
    """
    Create data directory
    """
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)


def get_csv_path(id):
    """
    from csv_ID to csv path

    Returns:
        str: path to the csv
    """
    csv_filename = "data_" + id + ".csv"
    return DATA_DIR + "/" + csv_filename
        
        
def save_csv(df):
    """
    Save dataframe to .cvs

    Args:
        df (DataFrame): data to save to .csv
        
    Returns:
        (str): unique ID
    """
    csv_id = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    df.to_csv(get_csv_path(csv_id), index = False)
    log.info("Data saved into file named : " + get_csv_path(csv_id))
    return csv_id



def delete_csv(csv_id):
    """
    Delete a .csv file

    Args:
        csv_id (str): csv file to delete
    """
    csv_path = get_csv_path(csv_id)
    if(os.path.exists(csv_path) and os.path.isfile(csv_path)):
        os.remove(csv_path)
        log.info("Deleted data file named : " + csv_path)
    else:
        log.error(csv_path + " not found")


def print_node_info():
    """
    Print node's name, rate and other info
    """
    print("Node name : " + NODE_NAME)
    print("Node rate : " + str(NODE_RATE) + "Hz")
    print("Node log : " + LOG_FOLDER + '/' + LOG_FILENAME)


def read_vars_name(filename):
    """
    Read variables name from txt

    Args:
        data (ped_motion): PoseArray data

    Returns:
        (list(str)): list containing variables name
        (list(str)): list containing variables name printable
    """
    log.info("Reading vars from : " + filename)

    # init empty vars_name list
    vars_name = list()
    vars_name_printable = list()

    # read .txt file
    with open(filename, 'r') as file:
        line = file.readline()
        while line:
            vars_name.append(line.strip())
            vars_name_printable.append(r'$' + str(line.strip()) + '$')
            line = file.readline()

    log.info("Causal analysis on variables : " + str(vars_name))
    vars_name.append('human_id')
    return vars_name, vars_name_printable
    

def distance(xa, ya, xb, yb):
    """
    Compute Euler distance between agent A and B

    Args:
        xa (float): x-coord A agent
        ya (float): y-coord A agent
        xb (float): x-coord B agent
        yb (float): y-coord B agent

    Returns:
        (float): Euler distance
    """
    d = sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
    return d


def bearing(xa, ya, xb, yb):
    """
    Compute angle between agent A and B

    Args:
        xa (float): x-coord A agent
        ya (float): y-coord A agent
        xb (float): x-coord B agent
        yb (float): y-coord B agent

    Returns:
        (float): angle between agent A and B
    """
    theta = atan2(yb - ya, xb - xa)
    return theta


def velocity(vx, vy):
    """
    Compute absolute velocity from vx and vy 

    Args:
        vx (float): human velocity along x-axis
        vy (float): human velocity along y-axis

    Returns:
        (float): absolute velocity of human
    """
    v = sqrt(vx ** 2 + vy ** 2)
    return v


def compute_causal_var(h_state, goal):
    """
    Compute new causal variable values from observed features

    Args:
        h_state (tuple[float]): _description_
        goal (tuple[float]): _description_

    Returns:
        dict[float]: New causal variable data
    """
    # Unzip tuples
    h_x, h_y, h_vx, h_vy = h_state
    goal_x, goal_y = goal

    # Compute new causal var values
    theta_g = bearing(h_x, h_y, goal_x, goal_y)
    d_g = distance(h_x, h_y, goal_x, goal_y)
    v = velocity(h_vx, h_vy)
    
    return {'theta_g' : theta_g, 'd_g' : d_g, 'v' : v}


def handle_obj_pose(data, selected_id):
    """
    new data on object position handler

    Args:
        data (SceneObjects): custom msg from MapServer

    Returns:
        (tuple[float]): position along x-axis, position along y-axis
    """
    obj = get_selected_obj(data, selected_id)

    x = obj.pose.pose.position.x
    y = obj.pose.pose.position.y
    log.debug("Object pose received : (x " + str(x) + " - y " + str(y) + ")")

    return (x, y)


def handle_human_pose(data, selected_id):
    """
    new data on human trajectory handler

    Args:
        data (Humans): custom msg from T2.5

    Returns:
        (tuple[float]): position along x-axis, position along y-axis, velocity along x-axis, velocity along y-axis
    """ 
    human = get_selected_human(data, selected_id)

    x = human.centroid.pose.position.x
    y = human.centroid.pose.position.y
    vx = human.velocity.twist.linear.x
    vy = human.velocity.twist.linear.y
    log.debug("Human pose received : (x " + str(x) + " - y " + str(y) + " - vx " + str(vx) + " - vy " + str(vy) + ")")

    return (x, y, vx, vy)


def selected_human_exist(data, selected_id):
    """
    return true if selected human is contained in the msg

    Args:
        data (Humans): custom msg from T2.5

    Returns:
        (bool): True if human id is contained in the msg. False otherwise
    """ 
    try:
        human = next(human for human in data.humans if human.id == selected_id)
    except:
        log.warn("human with id " + str(selected_id) + " not found")
        return False
    return True


def selected_obj_exist(data, selected_id):
    """
    return true if selected obj is contained in the msg

    Args:
        data (SceneObjects): custom msg from T2.5

    Returns:
        (bool): True if obj id is contained in the msg. False otherwise
    """ 
    try:
        obj = next(obj for obj in data.objects if obj.id == selected_id)
    except:
        log.warn("scene object with id " + str(selected_id) + " not found")
        return False
    return True


def get_selected_human(data, selected_id):
    """
    return select human msg

    Args:
        data (Humans): custom msg from T2.5

    Returns:
        (Human): selected human
    """ 
    human = next(human for human in data.humans if human.id == selected_id)
    return human


def get_selected_obj(data, selected_id):
    """
    return selected obj msg

    Args:
        data (Humans): custom msg from T2.5

    Returns:
        (SceneObject): selected obj
    """ 
    obj = next(obj for obj in data.objects if obj.id == selected_id)
    return obj


def print_causal_model_msg(cm: CausalModel):
    """
    build a string CausalModel msg

    Args:
        cm (CausalModel): CausalModel msg

    Returns:
        str: string to print descibing the CausalModel msg
    """
    string = "CausalModel message\n"
    string += "model : " + cm.model + "\n"
    string += "human_id : " + str(cm.human_id)
    return string