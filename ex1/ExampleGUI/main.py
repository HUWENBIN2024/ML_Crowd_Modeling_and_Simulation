from gui import MainGUI

import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser()
    cms_config = parser.add_argument_group("poison settings")

    cms_config.add_argument("--json_path", required=False, default="../scenarios/task4_bottleneck.json",
                    help="json file path.") 
    cms_config.add_argument("--iter", default=300, type=int, required=False,
                    help="set the number of iteration steps.") 
    cms_config.add_argument("--distance_mode", required=False, default='dijkstra',
                    help="config the distance mode. (the default one dijkstra )") 
    cms_config.add_argument("-:-r_max", default=2, type=int, required=False,
                    help="r_max for individual interaction.") 
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    gui = MainGUI(args)
    gui.start_gui()
