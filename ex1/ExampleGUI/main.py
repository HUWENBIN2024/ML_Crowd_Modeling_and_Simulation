from gui import MainGUI
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    cms_config = parser.add_argument_group("poison settings")

    cms_config.add_argument("--json_path", required=False, default="../scenarios/task4_chicktest.json",
                    help="json file path.") 
    cms_config.add_argument("--iter", default=100, type=int, required=False,
                    help="set the number of iteration steps.") 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gui = MainGUI(args)
    gui.start_gui()
