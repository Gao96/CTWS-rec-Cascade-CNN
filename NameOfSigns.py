
class SignNames:
    name=['changeLane','sharpTurn','reverseTurn','continuousTurn','steepSlope',
          'continuousSlopes','narrowRoad','narrowBridge','DoubleDirectionRoad',
          'pedestrian','children','domesticAnimal','wildAnimals',
          'signalLight','rockFall','crossWind','easyToSlip',
          'mountainRoad','damRoad','village','tunnel',
          'ferry','humpBridge','unevenRoad','waterPassage',
          'railway(watched)','railway(unwatched)','non-motorVehicle','Disabled',
          'easyToAccident','slow','obstacle','caution!',
          'roadConstruction','turnOnTheLightInTunnel','tidalDriveway','keepDistance',
          'separateRoad','convergenceInFront','intersection','obliqueRoad',
          'roundabout']

    def getName(self,a):
        return self.name[a]