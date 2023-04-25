from gui import MainGUI
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    poison_setting = parser.add_argument_group("poison settings")

    poison_setting.add_argument("--json_path", required=False, default="../scenarios/sc0.json",
                    help="json file path.") 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gui = MainGUI(args)
    gui.start_gui()
