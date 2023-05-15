import json
import os
import sys

def load_scenario(path):
    """
    Loads a json scenario file
    :param path: the path of the josn scenario file 
    :return: the dictionary containing all the scenario's attributes infomartaion
    """
    with open(path, 'r') as f:
        attributes = json.load(f)
        return attributes

def add_pedestrian(attributes):
    """
    Adds a pedestrian to the data
    :param data: data containing the information of the scenario
    :return: data with an added pedestrian
    """
    attributesPedestrian = {
        "attributes" : {
            "id" : 1,
            "shape" : {
                "x" : 0.0,
                "y" : 0.0,
                "width" : 1.0,
                "height" : 1.0,
                "type" : "RECTANGLE"
            },
            "visible" : True,
            "radius" : 0.2,
            "densityDependentSpeed" : False,
            "speedDistributionMean" : 1.34,
            "speedDistributionStandardDeviation" : 0.26,
            "minimumSpeed" : 0.5,
            "maximumSpeed" : 2.2,
            "acceleration" : 2.0,
            "footstepHistorySize" : 4,
            "searchRadius" : 1.0,
            "walkingDirectionSameIfAngleLessOrEqual" : 45.0,
            "walkingDirectionCalculation" : "BY_TARGET_CENTER"
        },
        "source" : None,
        "targetIds" : [ 2 ],
        "nextTargetListIndex" : 0,
        "isCurrentTargetAnAgent" : False,
        "position" : {
            "x" : 13,
            "y" : 2.5
        },
        "velocity" : {
            "x" : 0.0,
            "y" : 0.0
        },
        "freeFlowSpeed" : 0.9681588310983258,
        "followers" : [ ],
        "idAsTarget" : -1,
        "isChild" : False,
        "isLikelyInjured" : False,
        "psychologyStatus" : {
            "mostImportantStimulus" : None,
            "threatMemory" : {
                "allThreats" : [ ],
                "latestThreatUnhandled" : False
            },
            "selfCategory" : "TARGET_ORIENTED",
            "groupMembership" : "OUT_GROUP",
            "knowledgeBase" : {
                "knowledge" : [ ],
                "informationState" : "NO_INFORMATION"
            },
            "perceivedStimuli" : [ ],
            "nextPerceivedStimuli" : [ ]
        },
        "healthStatus" : None,
        "infectionStatus" : None,
        "groupIds" : [ ],
        "groupSizes" : [ ],
        "agentsInGroup" : [ ],
        "trajectory" : {
            "footSteps" : [ ]
        },
        "modelPedestrianMap" : { },
        "type" : "PEDESTRIAN"
    }
    attributes["name"] = "RiMEA Test 6 Add pedestraian "
    attributes["scenario"]["topography"]["dynamicElements"].append(attributesPedestrian)
    return attributes

def save_scenario(path, attributes):
    """
    Saves	the json scenario in the given path
    :param path: the output path of the scenario
           scenario: the scenario with the pedestrian added
    """
    with open(path, 'w') as f:
        json.dump(attributes, f, indent = 4)



path = "./RiMEA_Test_6.scenario"

attributes = load_scenario(path)

attributes = add_pedestrian(attributes)

output_path = "./RiMEA_Test_6_add_pedestrian.scenario"

save_scenario(output_path, attributes)