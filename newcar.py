# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)
# This code has again been hoisted by the CGS Digital Innovation Department
# giving credit to the above authors for the benfit of our education in ML

import math
import random
import sys
import os

import neat
import pygame

# Constants
# WIDTH = 1600
# HEIGHT = 880

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 30
CAR_SIZE_Y = 30

BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit

current_generation = 0  # Generation counter
"""
The Car Class 

Throughout this section, you will need to explore each function
and provide extenive comments in the spaces denoted by the 
triple quotes(block quotes) """ """.
Your comments should describe each function and what it is doing, 
why it is necessary and where it is being used in the rest of the program.

"""


class Car:
    """1. This Function:
    The ‘init’ function acts as the constructor for the car class which 
    is called upon when an object is created. As seen below, many attributes
    of the car are initialised. These attributes will be altered later in 
    the code to assign values and functions to each, however in this section
    ,each attribute is simply defined. It is necessary to define each attribute 
    in this section not only for readability but also because it adheres to 
    encapsulation. Encapsulation restricts direct access to the object's 
    internal state, only allowing controlled access through defined interfaces. 
    This protects the data, as it can only be modified through well defined 
    methods that ensure the data remains valid. These attributes will be used 
    in the following functions within the ‘Car’ class.
    """

    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load("car0.png").convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        # self.position = [690, 740] # Starting Position
        self.position = [830, 920]  # Starting Position
        self.angle = 0
        self.speed = 0

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [
            self.position[0] + CAR_SIZE_X / 2,
            self.position[1] + CAR_SIZE_Y / 2,
        ]  # Calculate Center

        self.radars = []  # List For Sensors / Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.alive = True  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed

    """ 2. This Function:
    The ‘draw’ function is defined which is responsible for updating the 
    car's visual representation on the screen. As seen below it draws the 
    sprite of the car on the screen which will be used later in the 
    run_simulation function to update the cars position. The draw function 
    is also used to draw the sensors that act as the input for the neural 
    network. It is not necessary to draw the sensors however as they would 
    still be present within the simulation but in drawing them it shows 
    the user exactly what the car is actually seeing and therefore the 
    input of the network.
    """

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        self.draw_radar(screen)  # OPTIONAL FOR SENSORS

    """ 3. This Function:
    The ‘draw_radar’ function is used to visually represent the radars 
    which act as the input for the neural network. This is once again 
    not entirely necessary for the actual purpose of the neural network 
    and machine learning but can be used to give the user insight into 
    what the input for the network is.
    """

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    """ 4. This Function:
    The ‘check_collision’ function is responsible for determining whether 
    the car has collided with the border of the map/track. This is done by 
    iterating through the corner of the car's sprite by using a ‘for’ loop 
    and checking whether a corner collides with the attribute defined as 
    ‘BORDER_COLOR’. If collision is detected it then sets 'self.alive' to 
    false which was an attribute created in the init function. This is 
    necessary for the purpose of the neural network and training the 
    program, with the check_collision function serving a crucial role 
    which provides feedback to the network and leads to better decision 
    making which improves the cars fitness over more and more generations. 
    The check_collision function is called within the update function 
    later in the car class.
    """

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    """ 5. This Function:
    The ‘check_radar’ function is responsible for creating the radars 
    that act as the car's way of viewing the map, acting as the five 
    input nodes for the NEAT algorithm. The radars extend from the 
    body of the car and go until a border is hit (defined by 
    BORDER_COLOR). The distance from the start of the radar (the body 
    of the car) and the end of the radar (where the radar hits the border) 
    is calculated and appended to the radars list which was created in 
    the init function. This data is saved and acts as the five input nodes 
    which influences the four output nodes (left, right, speed up, slow down) 
    which makes up the NEAT algorithm.
    """

    def check_radar(self, degree, game_map):
        length = 0
        x = int(
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + degree))) * length
        )
        y = int(
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + degree))) * length
        )

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + degree))) * length
            )
            y = int(
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + degree))) * length
            )

        # Calculate Distance To Border And Append To Radars List
        dist = int(
            math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2))
        )
        self.radars.append([(x, y), dist])

    """ 6. This Function:
    The ‘update’ function updates the cars position as it moves along the 
    map and many other factors that are required for the NEAT algorithm to 
    function. The function sets the speed for the first time which may then 
    be increased or decreased depending on the output node activated by the 
    radars (acting as the input nodes). The cars X and Y position is updated, 
    ensuring the doesn’t get close to the right and left edge as well as the
    top and bottom edge. The cars distance is updated as well as time 
    which are both factors that will be used to calculate the fitness of 
    each car. The fitness of the cars will be the decider on whether or 
    not a car should reproduce and pass on its genes. The new centre is 
    calculated and the four corners are calculated which is used in the 
    function check_collision to determine whether the car is alive. The 
    radars are then updated to check for obstacles using the check_radar 
    function. The update function is called within the ‘run simulation’ 
    function to update the cars position on the map and is crucial to the 
    NEAT process as it is what defines the cars position and subsequently 
    the radars position and data which acts as the input nodes for the 
    network.
    """

    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [
            int(self.position[0]) + CAR_SIZE_X / 2,
            int(self.position[1]) + CAR_SIZE_Y / 2,
        ]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length,
        ]
        right_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length,
        ]
        left_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length,
        ]
        right_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length,
        ]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    """ 7. This Function:
    The ‘get_data’ function retrieves the data obtained by the sensors and 
    normalises it, making it far more manageable and suitable to work as an 
    input for a neural network. A ‘for’ loop iterates over each radar 
    reading and normalises them by dividing them by thirty, this puts each 
    distance in a range of 1-10 (assuming 300 is the max distance of a radar). 
    The results are then put in the return_values list. This is an important 
    step as it normalises the data which acts as the input nodes for the 
    neural network. In normalising the data it can allow the program to 
    run more efficiently since it works with smaller numbers and therefore 
    a smaller range of data as the input for the neural network.
    """

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    """ 8. This Function:
    The ‘is_alive’ function simply returns the alive attribute of the car, 
    which acts as a boolean to determine whether or not the car has crashed. 
    The alive function is used later in the ‘run_simulation’ function to 
    determine whether the car's fitness should be increased and whether 
    the 'update' function should be used. The ‘is_alive’ function is 
    necessary since it affects the cars overall fitness (with fitness being 
    based on distance travelled which is obviously reliant on the car being 
    alive to travel any distance) and therefore the neural network which 
    decides whether or not to pass on a car’s genes through reproduction.
    """

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    """ 9. This Function:
    The ’get_award’ function is vital to the NEAT algorithm as it decides 
    which car should have its genomes passed onto the next generation, 
    influencing the decisions and behaviour of the next gen of cars. The 
    reward in this case is dependent on the distance that the car travels 
    which will incentivise the network to train cars for the purpose to 
    exclusively travel a long distance. The ‘get_reward’ function is used 
    in the ‘run_simulation’ function to update the reward for any cars that 
    are still alive.
    """

    def get_reward(self):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        return self.distance / (CAR_SIZE_X / 2)

    """ 10. This Function:
    The ‘rotate_center’ function rotates the car sprite by taking an image 
    as input (which is the sprite) and an angle. This is done by creating 
    a new rectangle which is rotated by the specified angle and then 
    matching the centre of the new rectangle with the original rectangle. 
    The original rectangle is then essentially cropped to match the new 
    dimensions of the rotated rectangle, finally the resulting rotated 
    image is returned. This function is essential to visualise the car's 
    orientation changes during the simulation so we can see the path the 
    car takes and its exact movements. The ‘rotate_center’ function is used 
    in the ‘update’ function to get the new rotated sprite and visualise 
    it on the screen.
    """

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


""" This Function:
The ‘run_simulation’ function is responsible for actually running the 
entire simulation:

1. Empty Collections:
Two empty collection lists are created which will be used to store each 
neural network (in ‘nets’) and each instance of the ‘car’ class (in ‘cars’)

2. Initialise PyGame and Display:
Pygame is initialised since the code requires the PyGame library to 
function. The display window which will appear on the machine that the 
code is being run on is created

3. For all Genomes Passed Create A New Neural Network:
For each genome a new neural network is created, using the config file to 
determine the properties of the neural network. Each network is added to 
the ‘nets’ list and has the genomes fitness set to 0. A new instance of 
the ‘Car’ class is created for each genome and will be controlled by the 
corresponding neural network.

4. Clock, Font, loading map, and generation:
The clock is created using PyGame and the fonts which will be used to 
display on screen text are defined. A map is also loaded from an image 
file which will be the track the cars race on. A counter is also set up to 
keep track of the current generation of cars.

5. Simple Counter and Quit Event:
A counter is set up which will be used to limit the time that the cars 
have to race each generation. A for loop is created which checks for the 
quit event (meaning the user wants to exit the simulation) and when the 
event occurs, the program is exited.

6. For Each Car get the Action:
A ‘for’ loop iterates over each car in the ‘cars’ list and the 
corresponding neural network is used to make a decision for the car. It 
begins with the neural network being activated with the input data that 
is controlled by the car's sensors (acting as the 5 input nodes). The 
neural network then makes a decision based on what it believes to be the 
most desirable action based on the input. This represents the four output 
nodes that are present within the neural network (left, right, speed up, 
slow down). 

7. If Car is Still Alive and Counter:
This section checks whether each car is still alive by using a ‘for’ loop 
and the ‘is_alive’ function which was defined within the ‘car’ class. If 
the car is still alive the ‘update’ function is used to update the cars 
position on the map and the fitness of the car is increased using the 
‘get_reward’ function which was also defined in the ‘car’ class. If no 
cars are still alive the loop breaks, ending the current simulation and 
starting a new generation of cars. A counter variable is created which 
is used to limit the time each generation of cars have to race.

8. Draw Map and All Cars:
A ‘for’ loop is created which draws each alive car on the screen using the 
‘draw’ function defined in the ‘car’ class as well as PyGames built in 
‘blit’ function which simply draws an image on another image (in this case 
the car sprites are drawn on the map).

9. Display Info and Framerate:
Text information is calculated and displayed on the screen such as the 
current generation and the number of alive cars. This is also done using 
PyGames' built in ‘blit’ function to display text on the game screen. The 
framerate is also regulated at 60FPS using PyGames ‘clock.tick’.
"""


def run_simulation(genomes, config):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Comic_Sans", 30)
    alive_font = pygame.font.SysFont("Comic_Sans", 20)
    mean_font = pygame.font.SysFont("Comic_Sans", 20)
    game_map = pygame.image.load("map2.png").convert()  # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2  # Slow Down
            else:
                car.speed += 2  # Speed Up

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:  # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = generation_font.render(
            "Generation: " + str(current_generation), True, (0, 0, 0)
        )
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)


        pygame.display.flip()
        clock.tick(60)  # 60 FPS


""" 1. This Section:
    This is the main section of the code and is the part that is actually 
    executed:

1. If __name__ == “__main”:
This line serves as an entry point for the script, only being executed if 
run directly rather than being imported as a module.

2. Load Config:
The config_path variable is set to the location of the config file and 
‘neat.config’ is used to create the config object. The ‘config’ object 
holds all the parameters present within the config file which alters the 
way the neural network and simulation will behave. These parameters fall 
into four categories: 

‘DefaultGenome’: Parameters that are related to the structure of the neural 
network and the representation of genomes, node activation functions, 
aggregation methods, bias options, compatibility measures, connection 
weights, and more.

‘DefaultReproduction’: Determines how reproduction of cars occurs: elitism 
(the number of top-performing genomes that are carried over to the next 
generation), and the survival threshold (the proportion of species that 
are allowed to reproduce).

‘DefaultSpeciesSet’: Determines how similar genomes must be to belong to 
the same species using the ‘compatability_threshold’.

‘DefaultStagnation’: Parameter regarding when a species is not making 
progress/not becoming fitter, how stagnation is detected, how many 
generations it takes before a species is considered stagnant, and how many 
species are protected from stagnation.

3. Population and Reporters:
A new population of genomes is created based on the configurations present 
in the config file. Two reporter objects are created which is used to print 
information to the terminal during the simulation and gather and calculate 
various statistics. The standard reporter is used to provide details about 
the best fitness for each generation and any other relevant information 
about the evolutionary process. The statistics reporter gathers and 
calculates various statistics about the population, tracking stats such 
as avg fitness, number of species and best fitness (as well as other 
relevant data).

4. Run Simulation For A Maximum of 1000 Generations:
The ‘run_simulation’ function is called for a maximum of 1000 generations, 
beginning the simulation process. 
"""
if __name__ == "__main__":
    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run Simulation For A Maximum of 1000 Generations
    population.run(run_simulation, 1000)
