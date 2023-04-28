# Excicise 1

Simple framework for the cellular automaton in exercise 1.

## installation

Please note that our program needs ***python>=3.10.0***, and the requirements.txt file is a little bit different from the original one. Please rerun it.

```
pip install -r requirements.txt 
```

## Simulation

```
python main.py --json_path <path of a json file> --iter <# of steps> --distance_mode <dijkstra or euclidean> --r_max <r max param>
```

example:

```
python main.py --json_path ../scenarios/sc0.json --iter 100 --distance_mode dijkstra --r_max 2
```

# Json

An example:
Coordinates are form of (x, y).

```
{
    "shape": [200, 200],
    "targets": [
                    [50, 20],
                    [180, 60]
                ],
    "pedestrians": [
                    [[31, 2]    , 2.3],
                    [[0, 0]     , 1.8],
                    [[199,59]   , 2.1]
                ],
    "obstacles": [
                    [20, 10],
                    [21, 10]
                 ]
            
}
```
