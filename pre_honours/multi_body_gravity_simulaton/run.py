from solar import Solar

if __name__ == "__main__":
    #take user input to determine which experiment to run 
    experiment = input("NOTE: Only one experiment can be done at a time. Please restart the program to do other experiments.\nPlease choose the experiment you would like to do from the below list:\n1. Run the orbit simulation with no experiment\n2. Orbital Periods\n3. Energy Conservation and Alternative Integration Methods\n4. Planetary Alignment\nEnter here: ")

    #initialize a solar object to the run the experiments with
    s = Solar()
    
    #parsing user input
    if experiment == "1":
        s.run()
    elif experiment == "2":
         s.experiment_1()
    elif experiment == "3":
        s.experiment_2()
    elif experiment == "4":
        s.experiment_3()
    
