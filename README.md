# master-thesis-code
Python code for railway disruption rescheduling problem in collaboration with the software Viriato from SMA and Partner AG

The code containts 11 python files. The main code runs through main.py. 

# Neighbourhood operators

Cancel, cancel from, return and emergency bus are implemented. 
Delay, delay from and emergency train are mostly implemented, but are not running on the main code. The feasibility check needs to be verified for those operators in order to be implemented in the main code.

# API

All the method calls from Python to Viriato are implemented in viriato_interface.py. 
