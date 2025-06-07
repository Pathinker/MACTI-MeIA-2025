import os
import numpy as np
import pandas as pd
from faker import Faker

REGISTERS = 10
HEADERS = ["Name", "Address", "Telephone", "ISR", "Month"]
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "October", "November", "December"]
LOWER_BOUND = 18837.38
UPPER_BOUND = 38775.38
ISR =  23.52
FAKE = Faker('en_US')

ISR_DATA_FRAME =  pd.DataFrame(columns=HEADERS)
ANUAL_ISR_DATA_FRAME =  pd.DataFrame(columns=HEADERS[:-1])

for i in range(REGISTERS):
    register_name = FAKE.name()
    register_address = FAKE.address()
    register_telephone = FAKE.phone_number()
    register_ISR = 0.0
    
    register_address = FAKE.address().replace("\n", " ")

    for j in range(len(MONTH)):
        income_salary = LOWER_BOUND + ((UPPER_BOUND - LOWER_BOUND) * np.random.rand())
        isr_tax = (income_salary - LOWER_BOUND) * (ISR / 100)
        ISR_DATA_FRAME.loc[len(ISR_DATA_FRAME)] = [register_name, register_address, register_telephone, isr_tax, MONTH[j]]
        register_ISR += isr_tax

    ANUAL_ISR_DATA_FRAME.loc[len(ANUAL_ISR_DATA_FRAME)] = [register_name, register_address, register_telephone, register_ISR]

ISR_DATA_FRAME.to_csv(os.getcwd() + "\\introduction\\ISR\\ISR.csv", index = False)
ANUAL_ISR_DATA_FRAME.to_csv(os.getcwd() +  "\\introduction\\ISR\\anual_ISR.csv", index = False)