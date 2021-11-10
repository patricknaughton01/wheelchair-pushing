import klampt

world = klampt.WorldModel()
world.loadFile("../Model/worlds/TRINA_world_cholera.xml")
wheelchair_model: klampt.RobotModel = world.robot("wheelchair")
for i in range(wheelchair_model.numLinks()):
    link: klampt.RobotModelLink = wheelchair_model.link(i)
    print(link.getName(), link.getTransform())
