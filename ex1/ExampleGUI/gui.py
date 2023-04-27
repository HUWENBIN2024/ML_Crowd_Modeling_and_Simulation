import sys
import tkinter
from tkinter import Button, Canvas, Menu
from scenario_elements import Scenario, Pedestrian
import time
import json
import numpy as np

def load_json(path):
    '''
    return a dictionary that map 'shape, targets, pedestrians, obstacles' to lists.

    args: 
        path: path of the json file you need to configure your scenario.
    '''
    f = open(path)
    config_dist = json.load(f)
    f.close()
    return config_dist
    

class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """

    def __init__(self, args):
        self.args = args
        self.sc = None

    def create_scenario(self, canvas, canvas_image, win):
        for i in range(self.args.iter):
            self.step_scenario(canvas, canvas_image)
            win.update()
            time.sleep(0.05)

    def restart_scenario(self, canvas, canvas_image):
        '''
        restarts a scenario.

        args :
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        '''
        self.load_scenario(canvas, canvas_image)

    def step_scenario(self, canvas, canvas_image):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): Add _description_
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        """
        self.sc.update_step()
        self.sc.to_image(canvas, canvas_image)

    def load_scenario(self, canvas, canvas_image):
        '''
        load a specific scenario described by a json file.

        args:
            path : path of the json file you need to configure your scenario.
            canvas (tkinter.Canvas): Add _description_.
            canvas_image (missing _type_): Add _description_.
        '''
        config_dist = load_json(self.args.json_path)
        sc = Scenario(config_dist['shape'][0], config_dist['shape'][0])

        targets = np.array((config_dist['targets'])).T
        try:
            obstacles = np.array((config_dist['obstacles'])).T
            sc.grid[obstacles[0], obstacles[1]] = sc.NAME2ID['OBSTACLE']
        except:
            pass
        sc.grid[targets[0], targets[1]] = sc.NAME2ID['TARGET']
        
        sc.recompute_target_distances()
        sc.pedestrians = [Pedestrian(p[0], p[1]) for p in config_dist['pedestrians']]
        self.sc = sc
        # scenario to image
        self.sc.to_image(canvas, canvas_image)

        return sc
    def visual_cost(self, canvas, canvas_image):
        self.sc.target_grid_to_image('cost', canvas, canvas_image)

    def exit_gui(self, ):
        """
        Close the GUI.
        """
        sys.exit()


    def start_gui(self, ):
        """
        Creates and shows a simple user interface with a menu and multiple buttons.
        Only one button works at the moment: "step simulation".
        Also creates a rudimentary, fixed Scenario instance with three Pedestrian instances and multiple targets.
        """
        # create an environment
        win = tkinter.Tk()
        win.geometry('800x800')  # setting the size of the window
        win.title('Cellular Automata GUI')

        # add a menu
        menu = Menu(win)
        win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='New', command=self.create_scenario)
        file_menu.add_command(label='Restart', command=self.restart_scenario)
        file_menu.add_command(label='Close', command=self.exit_gui)
        file_menu.add_command(label='load json files', command=self.load_scenario)

        # create a canvas
        canvas = Canvas(win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])  # creating the canvas
        canvas_image = canvas.create_image(5, 50, image=None, anchor=tkinter.NW)
        canvas.pack()

        # create a scenario configured by a json file
        self.load_scenario(canvas, canvas_image)

        btn = Button(win, text='Step simulation', command=lambda: self.step_scenario(canvas, canvas_image))
        btn.place(x=20, y=10)
        btn = Button(win, text='Restart simulation', command=lambda: self.restart_scenario(canvas, canvas_image))
        btn.place(x=200, y=10)
        btn = Button(win, text='Create simulation', command=lambda: self.create_scenario(canvas, canvas_image, win))
        btn.place(x=380, y=10)
        btn = Button(win, text='Cost Visualization', command=lambda: self.visual_cost(canvas, canvas_image, win))
        btn.place(x=550, y=10)

        win.mainloop()
